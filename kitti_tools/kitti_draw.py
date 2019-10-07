
import matplotlib.pyplot as plt
import numpy as np 



# from kitti_tools.kitti_draw import vis_geoDist
def vis_geoDist(img1_rgb, geo_dists, x1, 
                geo_dists_2=None, x2=None, mask=None, 
                show=False):
    # geo_dists = np.sqrt(utils_F._sym_epi_dist(F_gt_th, torch.from_numpy(x1[unique_rows_all_idxes]), torch.from_numpy(x2[unique_rows_all_idxes])).numpy())
    plt.hist(geo_dists, 100)
    plt.show()
    # geo_dists = np.clip(geo_dists, 0, 10.)
    # factor = 1/(geo_dists.max() + 1e-8)
    factor = 1
    dot_size = 1
    print("factor: ", factor)
    print("geo_dists: ", geo_dists.shape)
    # print("geo_dists norm: ", geo_dists*factor*dot_size)
    print("x1: ", x1.shape)
    # print("x1: ", x1)
    plt.figure(figsize=(30, 8))
    plt.imshow(img1_rgb)
    if mask is None:
        scatter = plt.scatter(x1[:, 0], x1[:, 1], s=geo_dists*factor*dot_size, c='r', edgecolors='w', linewidths=2.)  
        if x2 is not None and geo_dists_2 is not None:
            scatter =plt.scatter(x2[:, 0], x2[:, 1], s=geo_dists_2*factor*dot_size, c='b', edgecolors='w', linewidths=2.)  
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
        legend2 = plt.legend(handles, labels, loc="upper right", title="Sizes")
    else:
        plt.scatter(x1[mask, 0], x1[mask, 1], s=geo_dists[mask]*50, c='r', edgecolors='w', linewidths=2.)
    if show: plt.show()