import torch


class TriangularCausalMask:
    def __init__(self, B, L, device='cpu'):
        """

        :param B: batch size
        :param L: length of the vector
        :param device: default cpu
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """

        :return: the mask matrix (Bx1xLxL), up-triangular
        example for [[0, 1, 1],
                     [0, 0, 1],
                     [0, 0, 0]] if L = 3
        """
        return self._mask

