import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
  """
  return torch.cat([x.view(-1) for x in xs])


class Architect(object): # compute gradients of alphas

  def __init__(self, model, args):
    self.network_momentum = args.momentum # momentum for optimizer of theta
    self.network_weight_decay = args.weight_decay # weight decay for optimizer of theta
    self.model = model # main model with respect to theta and alpha
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), # optimizer for alpha
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer): # second order approximation
    """
    loss on train set and then update w_pi, not-in-place
      Args
        network_optimizer : optimizer of theta, not optimizer of alpha
    """
    loss = self.model._loss(input, target) #각각 input_train, target_train
    theta = _concat(self.model.parameters()).data # flatten current weights # theta: torch.Size([1930618])
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum) # fetch momentum data from theta optimizer
    except:
      moment = torch.zeros_like(theta)
    # flatten all gradients
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta # indeed, here we implement a simple SGD with momentum and weight decay # theta = theta - eta * (moment + weight decay + dtheta)
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled): # 여기서 학습을 진행
    self.optimizer.zero_grad() # alpha optimizer
    if unrolled: # second order
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else: # first order
        self._backward_step(input_valid, target_valid) # directly optimize alpha on w, instead of w_pi
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid): # first order 
    loss = self.model._loss(input_valid, target_valid) # 단순 validation loss 계산
    loss.backward() # autograd 를 사용하여 역전파 단계를 계산 # both alpha and theta require grad but only alpha optimizer will step in current phase.

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer): # first order approximation
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer) # w' = w - eta * grad(L_train) (one-step)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid) # w'에 대한 validaiton loss

    unrolled_loss.backward() # this will update w' model, but NOT w model
    dalpha = [v.grad for v in unrolled_model.arch_parameters()] # grad(L(w', a), a), part of Eq. 6
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train) # Eq.7에서 second term

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data) # g = g - eta * ig, from Eq. 7

    # update된 alpha를 self.model로 전달
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of .parameters() + fill_()
        theta : flatten weights, need to reshape to original shape
    """
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size()) # restore theta[] value to original shape
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v) # w+
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) # dalpha{L_train{w+, alpha}}

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v) # w-
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters()) # dalpha{L_train{w-, alpha}}

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)] # len: 2 h0 torch.Size([14, 8])

