import numpy as np

def show_matrix(cm, cm_labels, title, fn):
    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt
    nb_classes = len(cm_labels)
    # plot and save confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_normed=True, show_absolute=False, class_names=cm_labels)
    fig.set_size_inches(nb_classes, nb_classes)
    if title is not None:
        ax.set_title(title)
    plt.show()

def visualize_attention(attention, labels, target, fn, norm=False):
    import matplotlib.pyplot as plt
    def to_xy(id, r):
        angle = np.pi * 30 * id / 180
        _x = np.cos(angle) * r
        _y = np.sin(angle) * r
        return _x, _y

    if "Pelvis" in labels:
        joints = ["L_Foot", "L_LowLeg", "L_UpLeg", "Pelvis", "R_UpLeg", "R_LowLeg", "R_Foot"]
    else:
        joints = ["lfoot", "ltibia", "lfemur", "lowerback", "rfemur", "rtibia", "rfoot"]
    vis_names = ["l-foot", "l-tibia", "l-femur", "lowerback", "r-femur", "r-tibia", "r-foot"]
    r = 10
    x = []
    y = []
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, joint in enumerate(vis_names):
        angle = np.pi * 30 * i / 180
        x.append(np.cos(angle) * r)
        y.append(np.sin(angle) * r)
        ax.text(np.cos(angle)*(r+2)-i//2, np.sin(angle)*(r+1.5), joint, size=15)
    ax.scatter(x, y, s=50)
    
    tar_label_id = labels.index(target) # label id
    att = attention[tar_label_id]
    print(att)
    
    tar_id = joints.index(target) # coordinate id
    tar_x, tar_y = to_xy(tar_id, r)
    for i in range(7):
        if norm:
            partner_att = att[i] / np.amax(att)
        else:
            partner_att = att[i]
        partner_x, partner_y = to_xy(joints.index(labels[i]), r) # coordinate id
        ax.plot([tar_x, partner_x], [tar_y, partner_y], color="blue",
                alpha=partner_att*2.2, linewidth = partner_att*12)
        ax.text((tar_x+partner_x)/2-1, (tar_y+partner_y)/2,
                str(round(partner_att, 3)), size=10)
    
    ax.set_xlim(-13,13)
    ax.set_ylim(-0.5,11)
    ax.axis("off")
    plt.savefig(fn)
    

def make_confusion_matrix(preds, gts):
    from mlxtend.evaluate import confusion_matrix 
    cm = confusion_matrix(y_target=gts, y_predicted=preds, binary=False)
    return cm

def assign(preds): #(13, 13)
    from scipy.optimize import linear_sum_assignment
    cost = np.subtract(np.ones_like(preds), preds)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind

def assign_batch(preds_batch):
    batch_size, nb_imus = preds_batch.shape[:2]
    assigned = np.empty((batch_size, nb_imus))
    for i, preds in enumerate(preds_batch):
        assigned[i] = assign(preds)
    return np.reshape(assigned, (-1,))

def assign_batch_affinity(preds_batch, affinities):
    s_aff = preds_batch[0] # (nb_samples, 12, 8)
    s_cla = preds_batch[1] # (nb_samples, 12, 12)
    assigned = np.empty(s_aff.shape[:2])
    for idx, aff, cla in enumerate(zip(s_aff, s_cla)):
        aff_mat = np.zeros_like(cla)
        for i in cla.shape[0]:
            for j in cla.shape[1]:
                aff_mat[i,j] = aff[i, affinities[j]]
        scores = cla + aff_mat
        assigned[idx] = assign(scores)
    return np.reshape(assigned, (-1,))

def label2imu_idxs(imu_id, target_imus):
    return target_imus[imu_id]