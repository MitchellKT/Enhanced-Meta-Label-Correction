import torch
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call
from torch.autograd import forward_ad
import copy

########################### SINGLE STEP META #################################

def teacher_backward(rank,
                    main_net, main_opt,
                    teacher, teacher_opt,
                    enhancer, enhancer_opt,
                    data_s, target_s, 
                    data_g, target_g,
                    eta, num_classes,
                    injection_mode, mix_g,
                    **kwargs):
    
    # Teacher forward pass
    bs_s, bs_g = data_s.shape[0], data_g.shape[0]
    data_c = torch.cat([data_s, data_g])
    t_logit_c, t_repr_c = teacher(data_c, return_h=True)
    t_logit_s, t_logit_g = t_logit_c.split((bs_s, bs_g))
    t_repr_s, t_repr_g = t_repr_c.split((bs_s, bs_g))
    if mix_g:
        data_g_train, data_g_meta = data_g.chunk(2)
        target_g_train, target_g_meta = target_g.chunk(2) 
    
    # given current meta net, get corrected label
    pseudo_target_s = enhance_targets(t_logit_s, t_repr_s, target_s, enhancer, num_classes)


    # Update student
    if mix_g:
        old_main_net, loss_s = update_main_net(main_net, main_opt,
                                            data_s, pseudo_target_s.detach(),
                                            data_g_train, target_g_train)
    else:
        old_main_net, loss_s = update_main_net(main_net, main_opt,
                                            data_s, pseudo_target_s.detach())

    # Compute clean grad w.r.t student eval data
    if mix_g:
        gw, loss_g = compute_gw(main_net, main_opt, data_g_meta, target_g_meta)
    else:
        gw, loss_g = compute_gw(main_net, main_opt, data_g, target_g)

    # Compute jacobian vector product
    jvp = jacobian_vector_product(old_main_net, data_s, gw)

    # Meta loss
    reg_loss = meta_loss(jvp, pseudo_target_s, eta)
    
    # Supervised loss
    sup_loss = F.cross_entropy(t_logit_g, target_g)

    # Label retaining binary loss
    retain_loss = retain_conf_loss(enhancer, t_repr_g, t_logit_g,
                                    target_g, rank, num_classes,
                                    mode=injection_mode)

    # Update meta net
    t_loss = sup_loss + retain_loss + reg_loss
    teacher_opt.zero_grad()
    enhancer_opt.zero_grad()
    t_loss.backward()
    teacher_opt.step()
    enhancer_opt.step()

    return loss_g, loss_s, t_loss


########################### MULTI STEP META #################################

def teacher_backward_ms(rank, args,
                        main_net, main_opt,
                        teacher, teacher_opt,
                        enhancer, enhancer_opt,
                        data_s, target_s, 
                        data_g, target_g,
                        eta, num_classes,
                        injection_mode, mix_g):
    
    # Teacher forward pass
    bs_s, bs_g = data_s.shape[0], data_g.shape[0]
    data_c = torch.cat([data_s, data_g])
    t_logit_c, t_repr_c = teacher(data_c, return_h=True)
    t_logit_s, t_logit_g = t_logit_c.split((bs_s, bs_g))
    t_repr_s, t_repr_g = t_repr_c.split((bs_s, bs_g))
    if mix_g:
        data_g_train, data_g_meta = data_g.chunk(2)
        target_g_train, target_g_meta = target_g.chunk(2) 
    
    # given current meta net, get corrected label
    pseudo_target_s = enhance_targets(t_logit_s, t_repr_s, target_s, enhancer, num_classes)

    old_nets = []
    data_s_chunks = data_s.chunk(args.gradient_steps)
    pseudo_target_s_chunks = pseudo_target_s.chunk(args.gradient_steps)
    loss_s = torch.zeros(1).to(rank)
    
    for step in range(args.gradient_steps):

        # Fetch step data
        data_step = data_s_chunks[step]
        target_step = pseudo_target_s_chunks[step].detach()

        # Update student
        if mix_g:
            old_main_net, loss = update_main_net(main_net, main_opt,
                                                data_step, target_step,
                                                data_g_train, target_g_train)
        else:
            old_main_net, loss = update_main_net(main_net, main_opt,
                                                data_step, target_step)
        old_nets.append(old_main_net)
        loss_s += loss
    loss_s /= args.gradient_steps

    # compute gw for updating meta_net
    if mix_g:
        gw, loss_g = compute_gw(main_net, main_opt, data_g_meta, target_g_meta)
    else:
        gw, loss_g = compute_gw(main_net, main_opt, data_g, target_g)

    # Gather all jvps
    gamma = 1 - eta
    discount_factor = 1
    discounted_jvps = []
    for step in reversed(range(args.gradient_steps)):

        # Fetch step data
        data_step = data_s_chunks[step]
        model_step = old_nets[step]

        jvp = jacobian_vector_product(model_step, data_step, gw) # Compute jacobian vector product
        discounted_jvps.append(jvp.data * discount_factor)

        discount_factor *= gamma # Update the discount factor
    
    # Meta loss
    discounted_jvps = torch.cat(discounted_jvps)
    reg_loss = meta_loss(discounted_jvps, pseudo_target_s, eta)
    
    # Supervised loss
    sup_loss = F.cross_entropy(t_logit_g, target_g)

    # Label retaining binary loss
    retain_loss = retain_conf_loss(enhancer, t_repr_g, t_logit_g,
                                    target_g, rank, num_classes,
                                    mode=injection_mode)

    # Update meta net
    t_loss = sup_loss + retain_loss + reg_loss
    teacher_opt.zero_grad()
    enhancer_opt.zero_grad()
    t_loss.backward()
    teacher_opt.step()
    enhancer_opt.step()

    return loss_g, loss_s, t_loss

########################### Extra Utils #################################

def update_main_net(main_net, main_opt, data_s, pseudo_target_s, data_g_train=None, target_g_train=None):
    # a copy of the old student
    old_main_net = copy.deepcopy(main_net)

    if data_g_train is not None:
        # compute logits
        bs_s, bs_g = data_s.shape[0], data_g_train.shape[0]
        all_data = torch.cat([data_s, data_g_train])
        all_logits = main_net(all_data)
        logits_s, logits_g = all_logits.split((bs_s, bs_g))
        
        # compute loss for updating the student
        loss_s = F.cross_entropy(logits_s, pseudo_target_s)
        loss_g = F.cross_entropy(logits_g, target_g_train)
        loss = loss_s + loss_g
    else:
        logits = main_net(data_s)
        loss = F.cross_entropy(logits, pseudo_target_s)

    # Update main nets
    main_opt.zero_grad()
    loss.backward()
    main_opt.step()
    return old_main_net, loss

def enhance_targets(t_logit_s, t_repr_s, target_s, enhancer, num_classes):
    retain_conf = enhancer(t_repr_s, target_s)
    preds = F.softmax(t_logit_s, dim=1)
    one_hot = F.one_hot(target_s, num_classes)
    pseudo_target_s = retain_conf * one_hot + (1-retain_conf) * preds
    return pseudo_target_s

def compute_gw(main_net, main_opt, data_g, target_g):
    # compute gw for updating meta_net
    logit_g = main_net(data_g)
    loss_g = F.cross_entropy(logit_g, target_g)
    main_opt.zero_grad()
    loss_g.backward()
    gw = [param.grad.data for param in main_net.parameters()]
    # DONT DO OPTIMIZATION STEP
    return gw, loss_g

def meta_loss(jvp, pseudo_target_s, eta):
    # Meta loss
    batch_dot_product = (jvp.data * pseudo_target_s).sum(1)
    meta_loss = eta * batch_dot_product.mean() # Batch dot product
    return meta_loss

def retain_conf_loss(enhancer, t_repr_g, t_logit_g, target_g, rank, num_classes, mode='adversarial'):
    # Label retaining binary loss
    if mode == 'random':
        target_g_fake = torch.clone(target_g)
        target_g_fake[::2] = torch.randint_like(target_g_fake[::2], high=num_classes).to(rank)
    elif mode == 'adversarial':
        top_two_preds = torch.topk(t_logit_g, 2, dim=1, sorted=True)[1]
        adversarial_labels = torch.where(top_two_preds[:,0] != target_g, top_two_preds[:,0], top_two_preds[:,1])
        target_g_fake = torch.clone(target_g)
        target_g_fake[::2] = adversarial_labels[::2].to(rank)
    else:
        return 0
    target_g_mask = torch.eq(target_g_fake, target_g).type(torch.float).to(rank)
    retain_conf_g = enhancer(t_repr_g, target_g_fake)
    retain_conf_g = torch.clamp(retain_conf_g, min=0, max=1) # prevents floating point errors
    retain_conf_loss = F.binary_cross_entropy(retain_conf_g, target_g_mask.reshape_as(retain_conf_g))
    return retain_conf_loss

def jacobian_vector_product(model, inputs, vector, method='forward'):
    if method == 'forward':
        '''
        jvp products using forward mode AD as demonstrated in:
        https://pytorch.org/tutorials/intermediate/forward_ad_usage.html
        '''
        with torch.no_grad():
            with forward_ad.dual_level():
                params = {}
                if type(vector) is torch.Tensor:
                    offset = 0
                for i, (name, p) in enumerate(model.named_parameters()):
                    if type(vector) is torch.Tensor:
                        vector_part = vector[offset: offset+p.nelement()].view(p.size())
                        offset += p.nelement()
                        params[name] = forward_ad.make_dual(p, vector_part)
                    else:
                        params[name] = forward_ad.make_dual(p, vector[i])


                out = functional_call(model, params, inputs)
                lsm = F.log_softmax(out, dim=1)
                jvp = torch.autograd.forward_ad.unpack_dual(lsm).tangent
        return jvp
    elif method == 'double-back-trick':
        '''
        jvp products using double backward as demonstrated in:
        https://j-towns.github.io/2017/06/12/A-new-trick.html
        https://discuss.pytorch.org/t/forward-mode-ad-vs-double-grad-trick-for-jacobian-vector-product/159037
        '''
        out = model(inputs)
        lsm = F.log_softmax(out, dim=1)
        v = torch.zeros_like(lsm, requires_grad=True)
        g = torch.autograd.grad(lsm, model.parameters(), v, create_graph=True)
        jvp = torch.autograd.grad(g, v, vector)[0]
        return jvp.detach()
    else:
        raise NotImplementedError