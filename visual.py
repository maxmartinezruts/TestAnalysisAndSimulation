import scipy.io as sio
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import numpy


Tl = 0.2
TI = 0.2
tau_v = 0.3
tau_m = 0.3
kv = 1
km = 1
wm = 10
cm = 0.25
w = np.linspace(0,20, 100)
y = np.absolute(kv*(1+1j*Tl*w)/(1+1j*TI*w)*np.exp(-1j*w*tau_v)*wm**2/((1j*w)**2+2*cm*wm*1j*w+wm**2))
plt.plot(w, y)
plt.show()
import csv


TLs = np.linspace(0.1,1,3)
TIs = np.linspace(0.1,1,3)


def H(kv,Tl,TI,tau, wm, cm,w):
    return kv*(1+1j*Tl)/(1+1j*TI)*np.exp(-1j*w*tau)*wm**2/((1j*w)**2+2*cm*wm*1j*w+wm**2)


def get_visual_points(v_0,w):
    return v_0[0]*(1+1j*v_0[1]*w)/(1+1j*v_0[2]*w)*np.exp(-1j*w*v_0[3])*v_0[4]**2/((1j*w)**2+2*v_0[5]*v_0[4]*1j*w+v_0[4]**2)
def bode_plot(v_0, v_0_first):
    print(v_0_first)
    global error_v
    w_all = np.linspace(0, max(w), 200)

    visual_modeled_initial = get_visual_points(v_0_first, w_all)

    visual_modeled_all = get_visual_points(v_0, w_all)

    fig = plt.figure(figsize=(19, 12))

    H_pe_magnitude_all = np.absolute(visual_modeled_all)
    H_pe_angle_all = np.unwrap(np.angle(visual_modeled_all).flatten(), discont=-np.pi / 2)

    H_pe_magnitude_initial = np.absolute(visual_modeled_initial)
    H_pe_angle_initial = np.unwrap(np.angle(visual_modeled_initial).flatten(), discont=-np.pi / 2)

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax5 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax5.axis('off')

    params = np.array(['kv', 'TL', 'TI', 'tau_v', 'wm', 'cm'])
    initial = np.around(v_0_first, 4)
    current = np.around(v_0, 4)

    clust_data = np.transpose(np.array([params, initial, current]))
    collabel = ("Param", "Initial", "Current")
    table = ax5.table(cellText=clust_data, colLabels=collabel, loc='center', fontsize=20)
    table.scale(1, 4)

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    ax1.cla()
    ax1.semilogx(w_all, H_pe_angle_all)
    ax1.semilogx(w_all, H_pe_angle_initial)
    ax1.semilogx(w, phase_visual, 'rx')
    ax1.set_ylim([-10, 10])
    ax1.set_xlim([0, max(w)])
    ax1.grid(True, which="both")

    ax2.cla()
    ax2.loglog(w_all, H_pe_magnitude_all)
    ax2.loglog(w_all, H_pe_magnitude_initial)
    ax2.loglog(w, magnitudes_visual, 'rx')
    ax2.set_ylim([0, 10])
    ax2.set_xlim([0, max(w)])
    ax2.grid(True, which="both")

    fig.suptitle('#Iteration: ' + str(iteration) + ' J = ' + str(error_v) + '    V: ' + str(error_v), fontsize=20)

    fig.savefig(str(condition) + '_v/subject_' + str(subject)  + '.png')



def error(v_0):
    global iteration
    global errors
    global v_0_first
    global error_v

    visual_modeled = get_visual_points(v_0,w)

    error_v = 0
    for i in range(0,len(w)):
        error_v += (np.abs(visual_modeled[i]-visual_real[i])/np.abs(visual_real[i]))[0]**2
    error = error_v


    iteration += 1
    errors.append(error)
    # print(error[0],iteration)
    return error


conditions = ['C1','C2','C3','C4','C5','C6']
subject_ids = ['1','2','3','4','5','6',]

subjects = {}

for subject_id in subject_ids:
    subjects[subject_id] = {}
    file_name = 'ae2223I_measurement_data_subj' + subject_id + '.mat'
    subjects[subject_id]['file'] =file_name
    matlab_data = sio.loadmat(file_name)
    for condition in conditions:
        subjects[subject_id][condition] = {}
        subjects[subject_id][condition]['e'] = matlab_data['data_'+condition]['e'][0][0]
        subjects[subject_id][condition]['u'] = matlab_data['data_'+condition]['u'][0][0]
        subjects[subject_id][condition]['x'] = matlab_data['data_'+condition]['x'][0][0]
        subjects[subject_id][condition]['ft'] = matlab_data['data_'+condition]['ft'][0][0]
        subjects[subject_id][condition]['fd'] = matlab_data['data_'+condition]['fd'][0][0]
        subjects[subject_id][condition]['Hpe_FC'] = matlab_data['data_'+condition]['Hpe_FC'][0][0]
        subjects[subject_id][condition]['Hpxd_FC'] = matlab_data['data_'+condition]['Hpxd_FC'][0][0]
        subjects[subject_id][condition]['t'] = matlab_data['t'][0]
        subjects[subject_id][condition]['w_FC'] = matlab_data['w_FC']
        subjects[subject_id][condition]['H_magnitude_v'] = np.absolute(subjects[subject_id][condition]['Hpe_FC'][:,:])
        subjects[subject_id][condition]['H_angle_v'] = np.transpose(np.unwrap(np.transpose(np.angle(subjects[subject_id][condition]['Hpe_FC'][:,:])),discont=-np.pi))
        subjects[subject_id][condition]['H_magnitude_d'] = np.absolute(subjects[subject_id][condition]['Hpxd_FC'][:, :])
        subjects[subject_id][condition]['H_angle_d'] = np.transpose(np.unwrap(np.transpose(np.angle(subjects[subject_id][condition]['Hpxd_FC'][:, :])),discont=-np.pi))


        print(np.log(subjects[subject_id][condition]['Hpe_FC']))
        print('---------------')
kv_anova_visual = [['subj_id','magnitude','vehicle','motion']]
TL_anova_visual = [['subj_id','magnitude','vehicle','motion']]
TI_anova_visual = [['subj_id','magnitude','vehicle','motion']]
tau_v_anova_visual = [['subj_id','magnitude','vehicle','motion']]
wm_anova_visual = [['subj_id','magnitude','vehicle','motion']]
cm_anova_visual = [['subj_id','magnitude','vehicle','motion']]




for c_number in range(1,7):
    for s_number in range(1, 7):

        subject =  str(s_number)

        condition = 'C' + str(c_number)

        f_log = np.log(subjects[subject][condition]['Hpe_FC'])

        w = subjects[subject][condition]['w_FC']

        visual_real =  subjects[subject][condition]['Hpe_FC']
        motion_real =  subjects[subject][condition]['Hpxd_FC']

        phase_visual =      subjects[subject][condition]['H_angle_v']
        magnitudes_visual = subjects[subject][condition]['H_magnitude_v']
        phase_motion =      subjects[subject][condition]['H_angle_d']
        magnitudes_motion = subjects[subject][condition]['H_magnitude_d']

        params = np.array(['kv', 'TL', 'TI', 'tau_v', 'wm', 'cm'])


        v_0 = np.array([kv,Tl,TI,tau_v,wm,cm])
        v_0_first = v_0

        # path = str(condition) + '_v'
        #
        # for the_file in os.listdir(path):
        #     file_path = os.path.join(path, the_file)
        #     try:
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        #         if os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #         print(file_path)
        #     except Exception as e:
        #         print(e)

        min_errors = []
        K=0
        for TI in TIs:
            for TL in TLs:
                iteration = 0
                errors = []
                folder = str(condition)+'_v'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                v_0_first[1] = TL
                v_0_first[2] = TI
                v_0 = optimize.fmin(error,v_0_first)
                min_error = np.min(np.array(errors))
                min_errors.append(min_error)
                print(np.argmin(np.array(min_errors)), K, min_error, np.min(np.array(min_errors)), v_0)
                if np.argmin(np.array(min_errors)) == K:
                    # bode_plot(v_0, v_0_first)

                    kv_best =[subject, v_0[0], (c_number - 1)%3,  int((c_number-1)/3)]
                    TL_best =[subject, v_0[1], (c_number - 1)%3,  int((c_number-1)/3)]
                    TI_best =[subject, v_0[2], (c_number - 1)%3,  int((c_number-1)/3)]
                    tau_v_best =[subject, v_0[3], (c_number - 1)%3,  int((c_number-1)/3)]
                    wm_best =[subject, v_0[4], (c_number - 1)%3,  int((c_number-1)/3)]
                    cm_best =[subject, v_0[5], (c_number - 1)%3,  int((c_number-1)/3)]



                K+=1
        kv_anova_visual.append(kv_best)
        TL_anova_visual.append(TL_best)
        TI_anova_visual.append(TI_best)
        tau_v_anova_visual.append(tau_v_best)
        wm_anova_visual.append(wm_best)
        cm_anova_visual.append(cm_best)


print(kv_anova_visual)
with open('visual_anova_kv.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(kv_anova_visual)
with open('visual_anova_TL.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(TL_anova_visual)
with open('visual_anova_TI.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(TI_anova_visual)
with open('visual_anova_tau_v.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(tau_v_anova_visual)
with open('visual_anova_wm.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(wm_anova_visual)
with open('visual_anova_cm.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(cm_anova_visual)

csvFile.close()

