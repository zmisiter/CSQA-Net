import torch
import torch.nn as nn

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        self.conv_trans = nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding=0)
        self.flod = nn.Conv2d(18432, 2048, kernel_size=1, stride=1, padding=0)
        self.resize = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False)
        self.unfold = nn.Unfold(kernel_size=3, padding=1)
        self.l2_normalize = nn.functional.normalize

    def forward(self, part_ref, part_target):
        part_ref_unfold1 = self.unfold(part_ref)
        part_target_unfold = self.unfold(part_target)
        part_ref_unfold1 = part_ref_unfold1.view(part_ref_unfold1.size(0), part_ref_unfold1.size(1), -1)
        part_target_unfold = part_target_unfold.view(part_target_unfold.size(0), part_target_unfold.size(1), -1)
        part_ref_unfold = self.l2_normalize(part_ref_unfold1, dim=1)
        part_target_unfold = self.l2_normalize(part_target_unfold, dim=1)
        part_ref_unfold = part_ref_unfold.permute(0, 2, 1)
        R_part = torch.bmm(part_ref_unfold, part_target_unfold)
        max_value, max_index = torch.max(R_part, dim=1)
        part_ref_rerang_unfold = rerange(part_ref_unfold1, max_index)
        part_ref_rerang_unfold = part_ref_rerang_unfold.view(part_ref_rerang_unfold.size(0), part_ref_rerang_unfold.size(1), 7, 7)
        part_ref_rerang = self.flod(part_ref_rerang_unfold)

        part_res = self.conv_trans(torch.cat((part_ref_rerang, part_target), dim=1))
        mask = max_value.view(max_value.size(0), 1, part_ref_rerang.size(2), part_ref_rerang.size(3))
        part_res = part_res * mask
        part_res = part_res + part_target
        return part_res


def rerange(input, index):
    index = index.unsqueeze(1)
    index = index.expand(-1, input.shape[1], -1)
    output = torch.gather(input, 2, index)
    return output


if __name__ == '__main__':
    a = torch.randn(2,2048,7,7)
    b = torch.randn(2,2048,7,7)
    serach = SearchTransfer()
    c = serach(a, b)
    print(c.shape)
