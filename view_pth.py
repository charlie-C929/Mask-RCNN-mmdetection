import torch  # 命令行是逐行立即执行的
content = torch.load('modified_model.pth')
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['state_dict'].keys())
