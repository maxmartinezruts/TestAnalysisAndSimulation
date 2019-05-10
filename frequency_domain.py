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

TLs = np.linspace(0.1,1,5)
TIs = np.linspace(0.1,1,5)


def H(kv,Tl,TI,tau, wm, cm,w):
    return kv*(1+1j*Tl)/(1+1j*TI)*np.exp(-1j*w*tau)*wm**2/((1j*w)**2+2*cm*wm*1j*w+wm**2)


def get_visual_points(v_0, w):
    return v_0[0] * (1 + 1j * v_0[1] * w) / (1 + 1j * v_0[2] * w) * np.exp(-1j * w * v_0[4]) * v_0[6] ** 2 / (
                (1j * w) ** 2 + 2 * v_0[7] * v_0[6] * 1j * w + v_0[6] ** 2)


def get_motion_points(v_0, w):
    return v_0[3] * np.exp(-1j * w * v_0[5]) * v_0[6] ** 2 / (
                (1j * w) ** 2 + 2 * v_0[7] * v_0[6] * 1j * w + v_0[6] ** 2)
def bode_plot(v_0, v_0_first):
    w_all = np.linspace(0, max(w), 200)

    visual_modeled_initial = get_visual_points(v_0_first, w_all)
    motion_modeled_initial = get_motion_points(v_0_first, w_all)

    visual_modeled_all = get_visual_points(v_0, w_all)
    motion_modeled_all = get_motion_points(v_0, w_all)

    fig = plt.figure(figsize=(19, 12))

    H_pe_magnitude_all = np.absolute(visual_modeled_all)
    H_pe_angle_all = np.unwrap(np.angle(visual_modeled_all).flatten(), discont=-np.pi / 2)
    H_px_magnitude_all = np.absolute(motion_modeled_all)
    H_px_angle_all = np.unwrap(np.angle(motion_modeled_all).flatten(), discont=-np.pi / 2)

    H_pe_magnitude_initial = np.absolute(visual_modeled_initial)
    H_pe_angle_initial = np.unwrap(np.angle(visual_modeled_initial).flatten(), discont=-np.pi / 2)
    H_px_magnitude_initial = np.absolute(motion_modeled_initial)
    H_px_angle_initial = np.unwrap(np.angle(motion_modeled_initial).flatten(), discont=-np.pi / 2)

    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (0, 1))
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax5.axis('off')
    params = np.array(['kv', 'TL', 'TI', 'km', 'tau_v', 'tau_m', 'wm', 'cm'])
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

    # With motion
    ax3.cla()
    ax3.semilogx(w_all, H_px_angle_all)
    ax3.semilogx(w_all, H_px_angle_initial)
    ax3.semilogx(w, phase_motion, 'rx')
    ax3.set_ylim([-10, 10])
    ax3.set_xlim([0, max(w)])
    ax3.grid(True, which="both")

    ax4.cla()
    ax4.loglog(w_all, H_px_magnitude_all)
    ax4.loglog(w_all, H_px_magnitude_initial)
    ax4.loglog(w, magnitudes_motion, 'rx')
    ax4.set_ylim([0, 10])
    ax4.set_xlim([0, max(w)])
    ax4.set_xlabel('common xlabel')
    ax4.set_ylabel('common ylabel')
    ax4.grid(True, which="both")

    fig.suptitle('#Iteration: ' + str(iteration) + ' J = ' + str(error_v+error_m) + '    V: ' + str(error_v) + '   M: ' + str(error_m),fontsize=20)

    fig.savefig(str(condition) + '_m/subject_' + str(subject)  + '.png')


def error(v_0):
    global iteration
    global errors
    global v_0_first
    global error_v
    global error_m


    visual_modeled = get_visual_points(v_0, w)
    motion_modeled = get_motion_points(v_0, w)

    error_v = 0
    error_m = 0
    for i in range(2, len(w)):
        error_v += (np.abs(visual_modeled[i] - visual_real[i]) / np.abs(visual_real[i]))[0] ** 2
        error_m += (np.abs(motion_modeled[i] - motion_real[i]) / np.abs(motion_real[i]))[0] ** 2
    error = error_v + error_m

    iteration += 1
    errors.append(error)
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



for s_number in range(1,7):
    for c_number in range(4,7):

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


        v_0 = np.array([kv,Tl,TI,km,tau_v, tau_m,wm,cm])
        v_0_first = v_0

        path = str(condition)+str('_m')

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
                folder = str(condition)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                v_0_first[1] = TL
                v_0_first[2] = TI
                v_0 = optimize.fmin(error,v_0_first)
                min_error = np.min(np.array(errors))
                min_errors.append(min_error)
                print(np.argmin(np.array(min_errors)), K, min_error, np.min(np.array(min_errors)), v_0_first)

                if np.min(np.array(min_errors)) == min_error:
                    bode_plot(v_0, v_0_first)
                    #
                    # kv_read = np.zeros((6, 3))
                    # TL_read = np.zeros((6, 3))
                    # TI_read = np.zeros((6, 3))
                    # km_read = np.zeros((6, 3))
                    # tau_v_read = np.zeros((6, 3))
                    # tau_m_read = np.zeros((6, 3))
                    # wm_read = np.zeros((6, 3))
                    # cm_read = np.zeros((6, 3))
                    kv_read = numpy.genfromtxt('motion_kv.csv', delimiter=',')
                    TL_read = numpy.genfromtxt('motion_TL.csv', delimiter=',')
                    TI_read = numpy.genfromtxt('motion_TI.csv', delimiter=',')
                    km_read = numpy.genfromtxt('motion_km.csv', delimiter=',')
                    tau_v_read = numpy.genfromtxt('motion_tau_v.csv', delimiter=',')
                    tau_m_read = numpy.genfromtxt('motion_tau_m.csv', delimiter=',')
                    wm_read = numpy.genfromtxt('motion_wm.csv', delimiter=',')
                    cm_read = numpy.genfromtxt('motion_cm.csv', delimiter=',')
                    #
                    kv_read[s_number - 1, c_number - 4] = v_0[0]
                    TL_read[s_number - 1, c_number - 4] = v_0[1]
                    TI_read[s_number - 1, c_number - 4] = v_0[2]
                    km_read[s_number - 1, c_number - 4] = v_0[3]
                    tau_v_read[s_number - 1, c_number - 4] = v_0[4]
                    tau_m_read[s_number - 1, c_number - 4] = v_0[4]
                    wm_read[s_number - 1, c_number - 4] = v_0[6]
                    cm_read[s_number - 1, c_number - 4] = v_0[7]

                    numpy.savetxt("motion_kv.csv", numpy.asarray(kv_read), delimiter=",")
                    numpy.savetxt("motion_TL.csv", numpy.asarray(TL_read), delimiter=",")
                    numpy.savetxt("motion_TI.csv", numpy.asarray(TI_read), delimiter=",")
                    numpy.savetxt("motion_km.csv", numpy.asarray(km_read), delimiter=",")
                    numpy.savetxt("motion_tau_v.csv", numpy.asarray(tau_v_read), delimiter=",")
                    numpy.savetxt("motion_tau_m.csv", numpy.asarray(tau_m_read), delimiter=",")
                    numpy.savetxt("motion_wm.csv", numpy.asarray(wm_read), delimiter=",")
                    numpy.savetxt("motion_cm.csv", numpy.asarray(cm_read), delimiter=",")

                K += 1






