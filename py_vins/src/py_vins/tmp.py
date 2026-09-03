import numpy as np
from navlie import randvec
import random

random.seed(1)

anc_0 = np.array([3.273827392578125, 3.46404736328125, 1.8093309326171875])
anc_1 = np.array([3.186386962890625, 0.27394485473632812, 1.5884853515625])
anc_2 = np.array([2.850500244140625, -2.923056884765625, 1.89742041015625])
anc_3 = np.array([-2.497634521484375, -3.5018203125, 1.7730911865234375])
anc_4 = np.array([-2.95793310546875, 0.6128419189453125, 1.65714208984375])
anc_5 = np.array([-2.734676513671875, 3.65854248046875, 1.890254638671875])

anc_cov_0 = [0.02,0.02,0.09]
anc_cov_1 = [0.01,0.02,0.08]
anc_cov_2 = [0.01,0.02,0.05]
anc_cov_3 = [0.02,0.01,0.08]
anc_cov_4 = [0.01,0.02,0.05]
anc_cov_5 = [0.02,0.02,0.07]


# anc_cov_0 = [0.09,0.07,0.3]
# anc_cov_1 = [0.04,0.05,0.2]
# anc_cov_2 = [0.02,0.04,0.1]
# anc_cov_3 = [0.05,0.05,0.17]
# anc_cov_4 = [0.02,0.04,0.14]
# anc_cov_5 = [0.05,0.06,0.18]

anchor_list_cov = [anc_cov_0, anc_cov_1, anc_cov_2, anc_cov_3, anc_cov_4, anc_cov_5]
anchor_list = [anc_0, anc_1, anc_2, anc_3, anc_4, anc_5]

count = 0
for anc, cov in zip(anchor_list, anchor_list_cov):
    cov_mat = np.diag(cov)
    perturb = randvec(cov_mat)
    new_anc = anc.reshape(-1, 1) + perturb
    print("anchor" + str(count), ": ", new_anc.reshape(3,).tolist())
    count += 1