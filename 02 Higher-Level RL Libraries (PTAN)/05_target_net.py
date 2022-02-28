#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# The PTAN library——The TargetNet class
# TargetNet允许我们同步具有相同架构的两个网络，其目的是为了提高训练稳定性
# https://www.cnblogs.com/kailugaji/
import ptan
import torch.nn as nn

# 创建网络
class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.ff = nn.Linear(5, 3) # in_features=5, out_features=3, 权重大小：(3, 5)

    def forward(self, x):
        return self.ff(x)


if __name__ == "__main__":
    net = DQNNet()
    print("原网络架构：\n", net)
    tgt_net = ptan.agent.TargetNet(net)
    print("原网络权重：", net.ff.weight)
    print("目标网络权重：", tgt_net.target_model.ff.weight)
    # 上述原网络与目标网络权重相同

    # 然而，它们彼此独立，只是拥有相同的架构：
    net.ff.weight.data += 1.0
    print("-------------------------------------------------------------------")
    print("更新后：")
    print("原网络权重：", net.ff.weight)
    print("目标网络权重：", tgt_net.target_model.ff.weight)

    # 要再次同步它们，可以使用sync()方法
    tgt_net.sync() # weights from the source network are copied into the target network
    print("-------------------------------------------------------------------")
    print("同步后：")
    print("原网络权重：", net.ff.weight)
    print("目标网络权重：", tgt_net.target_model.ff.weight)
