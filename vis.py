# Charles Lett Jr.
# REU 2021
# this is called in main if enabled; creates graphs based on the inputted data
# ***currently broken and should not be used***



import matplotlib.pyplot as plt
from timeit import default_timer as rtimer
import numpy as np
import main

#----------------------------------------------------------------------------------------------------------
# begin run
# 'os.system' only works in command prompt (maybe linux terminal)
#            it clears the command window each run for readability;
# os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print("\n *** Running 'vis.py' *** ")

runtime_start = rtimer()
#----------------------------------------------------------------------------------------------------------
print('>> [VISUALIZER]')
# init class/profile index
init_class_ind = [i for i in range(len(main.class_profile))]
init_profile_ind = [len(main.class_profile[j]) for j in range(len(main.class_profile))]

# trained class/profile index
trained_class_ind = [i for i in range(len(main.upd_class_profile))]
trained_profile_ind = [len(main.upd_class_profile[j]) for j in range(len(main.upd_class_profile))]

def profile_plt():
    def test():
        x = []
        fig, ax = plt.subplots(main.anchor_count, 1)
        inc = 0
        for i in range(main.anchor_count):
            ax[i].set_title(('Anchor', i + 1))
            if i + 1 != main.anchor_count:
                ax[i].axes.get_xaxis().set_visible(False)
            ax[i].plot(x, main.anchor[:, 1])

        plt.subplots_adjust(hspace=0.4)
        fig.show()

    def anc_dist():
        # bar graph settings
        index = np.arange(main.anchor_count)
        # print(index)

        fig = plt.plot(1, 2)
        bar_width = 0.55
        color = ['blue', 'green']

        # data for graph
        titles = ['Class ' + str(init_class_ind[i] + 1) for i in init_class_ind]
        labels = [[] for i in range(len(init_class_ind))] # ticks: profiles
        init_data = [[] for j in range(len(init_class_ind))] # bar height
        train_data = [[] for j in range(len(trained_class_ind))] # bar height

        # add anchor labels to labels array
        for i in init_class_ind:
            for j in range(init_profile_ind[i]):
                # print(i, j)
                labels[i].append('Profile ' + str(j + 1))
                init_data[i].append(main.class_profile[i][j][0])
                train_data[i].append((main.upd_class_profile[i][j][0]))

        # for i in range(len(titles)):
            # print('INIT', titles[i], labels[i], init_data[i])
            # print('TRAIN', titles[i], labels[i], train_data[i])

        # print(j, k, l)
        ax[0].set_title(titles[0])
        ax[0].set_ylim(0, 1)
        ax[0].bar(labels[0],
                  init_data[0][0][0],
                  bar_width,
                  align='center')

        fig.tight_layout()
        # plt.legend()
        plt.show()

    anc_dist()
profile_plt()


#----------------------------------------------------------------------------------------------------------
runtime_end = rtimer()
print('>> Runtime (vis.py):', round(runtime_end - runtime_start, 2), 'seconds <<')

# end run
print(" *** 'vis.py' Finished. *** ")
# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
#----------------------------------------------------------------------------------------------------------
