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