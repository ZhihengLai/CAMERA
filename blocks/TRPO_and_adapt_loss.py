import torch
from torch.distributions.normal import Normal
import numpy as np
import wandb
import torchopt

from torch.nn.utils import parameters_to_vector ,vector_to_parameters
from maml_agent import New_maml_agent


#################---------   ADAPTATION LOSS  ---------###############

def policy_loss_for_adaptation(agent,buffer):
    '''Computes loss for the adaptation steps in the inner loops  '''

    _ , newlogprob, entropy = agent.get_action(buffer.observations, buffer.actions) 

    normalize_advantage=True
    if normalize_advantage==True:
        buffer.advantages= (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    pg_loss = -(newlogprob * buffer.advantages).mean()

    return pg_loss





############################################################################


#################---------   MAML AGENT TRPO UPDATE  ---------###############
    

def combine_gradients(gradient_list):
    """Combines gradients from a list by averaging them for each parameter.

    Args:
        gradient_list: A list of gradients, where each element is a list of
            tensors containing gradients for a set of parameters.

    Returns:
        A list of tensors containing the combined gradients, where each tensor
        has the same shape as the corresponding tensor in the input gradients.
    """

    combined_gradients = []

    for gradients_per_param in zip(*gradient_list):
        combined_gradient = torch.mean(torch.stack(gradients_per_param), dim=0)
        combined_gradients.append(combined_gradient)

    return combined_gradients

#----

def surrogate_loss(agent,buffer,old_distribution=None,logs_dict=None):
    

    _ , newlogprob, entropy, distribution = agent.get_action(buffer.observations, buffer.actions ,return_distribution=True) 

    normalize_advantage=True
    if normalize_advantage==True:
        buffer.advantages= (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    if old_distribution==None:
        if isinstance(distribution, Normal):
            old_distribution = Normal(loc=distribution.loc.detach(), scale=distribution.scale.detach())


    logratio = newlogprob - old_distribution.log_prob(buffer.actions).sum(1)
    ratio = logratio.exp()

    loss=  -(ratio * buffer.advantages).mean()

    kl=torch.distributions.kl.kl_divergence(distribution, old_distribution).mean()

    if logs_dict:
        logs_dict['policy gradient loss'].append(loss.item())
        logs_dict['entropy'].append(entropy.mean().item())
        logs_dict['aprox KL'].append(kl.item())
        
    return loss ,kl ,old_distribution


#----

def hessian_vector_product(kls, adapted_policies_states, maml_agent, damping=1e-2):

    kls=kls
    adapted_policies_states=adapted_policies_states
    maml_actor=maml_agent.actor
    
    def _hv_product(vector,retain_graph=True):
        grads_grad_kl_v=[]
        for kl, adapted_policy_states in zip(kls,adapted_policies_states):
            torchopt.recover_state_dict(maml_actor, adapted_policy_states)

            kl_grad = torch.autograd.grad(kl, maml_actor.parameters(),
                                                create_graph=True)
            flat_kl_grad = parameters_to_vector(kl_grad)
            
            #_product
            grad_kl_v = torch.dot(flat_kl_grad, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                            maml_actor.parameters(),
                                            retain_graph=retain_graph)
            grads_grad_kl_v.append(grad2s)

        grad_grad_kl_v = combine_gradients(grads_grad_kl_v) #average the hessian vector product over all adapted policies
        flat_grads_grad_kl_v = parameters_to_vector( grad_grad_kl_v )

        return flat_grads_grad_kl_v + damping * vector
    
    return _hv_product


#----

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


#-----

def maml_actor_trpo_update(maml_agent, data_buffers, adapted_policies_states,  config, logging=True):

    base_policy_state_dict = torchopt.extract_state_dict(maml_agent) #extract the state of the base maml agent (params before being adapted to a specific env)

    logs_dict={'policy gradient loss':[] ,'entropy' :[]  ,'aprox KL':[]}

    old_loss=0
    kls=[]
    surrogate_loss_gradients=[]
    old_distributions=[]

    #obtain surrogate losses (loss of adapted policies) , the kl divergences and save the probability distribution over actions of each adapted policy for the states in which it collected data
    for buffer ,adapted_policy_state in zip(data_buffers ,adapted_policies_states):
        torchopt.recover_state_dict(maml_agent, adapted_policy_state)
        
        # Compute the surrogate loss and calculate gradients with respect to each adapted policy
        adapted_policy_losses , kl ,distribution = surrogate_loss(agent=maml_agent,buffer=buffer) 

        surrogate_loss_gradients.append( torch.autograd.grad(adapted_policy_losses,
                                            maml_agent.actor.parameters(), # 注意这里只对actor的参数求梯度
                                            retain_graph=True))

        old_loss+=adapted_policy_losses
        kls.append(kl)
        old_distributions.append(distribution)

    # Loss accumulated by adding surrogate loss of every adapted policies together
    old_loss=old_loss/len(data_buffers)

    # Combine gradients of every adapted policies together
    policy_loss_grads=combine_gradients(surrogate_loss_gradients)
    flat_policy_loss_grads=parameters_to_vector(policy_loss_grads)

    #save old parameters of base policy
    torchopt.recover_state_dict(maml_agent, base_policy_state_dict)
    old_base_params = parameters_to_vector(maml_agent.parameters())
    old_base_actor_params = parameters_to_vector(maml_agent.actor.parameters())

    # Save the old parameters of each adapted policy
    old_params=[]
    old_actor_params=[]
    for adapted_policy_state in adapted_policies_states:
        torchopt.recover_state_dict(maml_agent, adapted_policy_state)
        vec_params = parameters_to_vector(maml_agent.parameters())
        actor_vec_params = parameters_to_vector(maml_agent.actor.parameters())
        old_params.append(vec_params)
        old_actor_params.append(actor_vec_params)
    

    #compute the direction of the update with conjugate gradient
    hess_vec_product = hessian_vector_product(kls,adapted_policies_states,maml_agent,damping=config.maml_TRPO["cg_damping"]) # The return is a function whose input is a vector.
    # stepdir: \hat{x}_k
    stepdir = conjugate_gradient(hess_vec_product,
                                        flat_policy_loss_grads, # accumulated gradients
                                        cg_iters=config.maml_TRPO["cg_iters"])

    # Compute the Lagrange multiplier
    shs = 0.5 * torch.dot(stepdir,
                            hess_vec_product(stepdir,retain_graph=False))
    # lagrange_multiplier: 1 / \sqrt{\frac{2*\delta}{\hat{x}_k^T H \hat{x}_k}}
    lagrange_multiplier = torch.sqrt(shs / config.maml_TRPO['max_kl'])

    step = stepdir / lagrange_multiplier




    ## Line search to find how much the parameters of the base policy are moved in the update direction
    with torch.no_grad():
        step_size = 3.0

        line_search_succeeded=False

        for _ in range(config.maml_TRPO['line_search_max_steps']):
            surrogate_policy_loss=0
            kl=0

            for old_adapted_policy_params, old_adapted_actor_params, buffer, old_distribution in zip(old_params, old_actor_params ,data_buffers, old_distributions):

                vector_to_parameters(old_adapted_policy_params, maml_agent.parameters()) # 把旧版本的全部参数load进agent，其中actor的参数会被用于line search

                updated_actor_params= old_adapted_actor_params - step_size * step # 由于step是针对maml_agent.actor的，需要对maml_agent.actor进行尝试
                vector_to_parameters(updated_actor_params , maml_agent.actor.parameters())

                s_loss , kl_div ,_= surrogate_loss(buffer=buffer,agent=maml_agent,old_distribution=old_distribution ,logs_dict=logs_dict)

                surrogate_policy_loss+=s_loss
                kl+=kl_div

            # The surrogate loss and KL divergence are averaged across all tasks
            surrogate_policy_loss=surrogate_policy_loss/len(data_buffers)
            kl=kl/len(data_buffers)

            # Check if the proposed update satisfies the constraints that
            # we improve with respect to the surrogate policy objective while also staying close enough (in term of kl div) to the old policy
            if (surrogate_policy_loss<old_loss) and (kl.item()<config.maml_TRPO['max_kl']):
                line_search_succeeded=True

                vector_to_parameters(old_base_params, maml_agent.parameters())
                updated_actor_params= old_base_actor_params - step_size * step # if the averaged kl and surrogate_policy_loss satisfy the constraint, directly apply the step_size on old_base_params, which is the original parameter.
                vector_to_parameters(updated_actor_params , maml_agent.actor.parameters())
                break
            
            else:
                ## Reduce step size if line-search wasn't successful
                step_size *= config.maml_TRPO["line_search_backtrack_ratio"]
                logs_dict={'policy gradient loss':[] ,'entropy' :[]  ,'aprox KL':[]}

        if line_search_succeeded==False:
            # If the line-search wasn't successful  revert to the original parameters
            vector_to_parameters(old_base_params, maml_agent.parameters())
        

    if logging==True:
        wandb.log({ 'maml policy gradient loss': np.array(logs_dict['policy gradient loss']).mean(),
                    'maml entropy ': np.array(logs_dict['entropy']).mean(),
                    'maml aprox KL ' :  np.array(logs_dict['aprox KL']).mean() } , commit=False )
        
def maml_critic_update(maml_agent:New_maml_agent, optimizer, meta_batch_size):
    for p in maml_agent.policy.critic_original.parameters():
        p.grad.data.mul_(1.0 / meta_batch_size)
    optimizer.step()

def maml_extractor_update(maml_agent:New_maml_agent, optimizer, meta_batch_size):
    maml_extractor = maml_agent.extractor.ofe_net_original
    for p in maml_extractor.parameters():
        p.grad.data.mul_(1.0 / meta_batch_size)
    optimizer.step()

    


