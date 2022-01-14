# search cell 단계
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  # mixed operation 정의해주는 부분
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList() #operation을 modulelist에 포함
    for primitive in PRIMITIVES: # 모든 후보 연산을 포함시킴
      op = OPS[primitive](C, stride, False) # op : layer들의 집합
      if 'pool' in primitive: #pooling이면 뒤에 batch normalization 시켜줌
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)


  def forward(self, x, weights): #mixed operation 계산(공식에 따라, 이때 weight = softmax(alpha))
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    # Create a searchable cell representing multiple architectures.
    # single architecture에서 models.py에 있는 Cell과 같음
    '''
        Args
            steps: The number of primitive operations in the cell, cell 내의 layer의 개수
            multiplier: The rate at which the number of channels increases.
            C_prev_prev: C_out[k-2]
            C_prev : C_out[k-1]
            C : C_in[k] (current)
            reduction_prev: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
    '''
    super(Cell, self).__init__()
    self.reduction = reduction

    # If previous cell is reduction cell, current input size does not match with
    # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0) # c[k-2]
    s1 = self.preprocess1(s1) # c[k-1]

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1) # height x width x depth를 1 x 1 x depth로 바꿔줌
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas() # alpha initialization

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input) # 2개의 input node
    for i, cell in enumerate(self.cells): # i는 cell의 index, cell은 그 cell, 여기서 weight 정의 (weight = softmax(alpha))
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1) # flatten을 시켜줌
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target): # 일반적으로 모델 안에 로스를 집어넣지는 않는다. 모델과 로스는 종속된 개념이 아니므로 분리하는 것이 좋지만, unrolled gradient 를 계산할 때 로스를 여러번 계산해야 하기 때문에 코드의 중복을 줄이기 위해서 넣어주었다.
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self): 
    k = sum(1 for i in range(self._steps) for n in range(2+i)) # self._steps=4이면 k가 14가나옴(2+3+4+5)
    num_ops = len(PRIMITIVES) # number of operations (여기선 8)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) # 14 X 8 배열
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [ # normnal cell과 reduction cell의 알파를 각각 저장
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights): # weights : [14, 8]
      gene = []
      n = 2
      start = 0
      for i in range(self._steps): # for each node
        end = start + n
        W = weights[start:end].copy() # [2, 8], [3, 8], ...
        edges = sorted(range(i + 2), # i+2 is the number of connection for node i
          key=lambda x: -max(W[x][k] # by descending order
                        for k in range(len(W[x])) # get strongest oerations 
                                  if k != PRIMITIVES.index('none')))[:2] # only has two inputs
        for j in edges: # for every input nodes j of current node i
          k_best = None
          for k in range(len(W[j])): # get strongest ops for current input j->i
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j)) # save ops and input node
        start = end
        n += 1 
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
"""
    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
"""

