#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import os
import shutil 

sure = input("Are you sure? Y/N: ")
if sure == 'Y':
    print('removing all files')
    shutil.rmtree('./data_ckpt', ignore_errors=True)
    shutil.rmtree('./eval.log', ignore_errors=True)
    shutil.rmtree('./test.log', ignore_errors=True)
    shutil.rmtree('./train.log', ignore_errors=True)
    shutil.rmtree('./latest_epoch_tested', ignore_errors=True)
else:
    print('Nothing removed!')