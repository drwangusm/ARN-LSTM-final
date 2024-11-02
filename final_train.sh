#!/bin/bash
# nohup bash final_train.sh > final_train.log 2>&1&  #服务器后台运行，log保存至 final_train.log中

# NTU-V1/V2
# ARN_joint
python3 src/train.py.py ARN_joint configs/NTU-V1/no-lstm/ARN_joint.cfg NTU -n 5 -v 2
python3 src/train.py.py ARN_joint configs/NTU-V2/no-lstm/ARN_joint.cfg NTU-V2 -n 5 -v 2
# ARN_temp
python3 src/train.py.py ARN_temp configs/NTU-V1/no-lstm/ARN_temp.cfg NTU -n 5 -v 2
python3 src/train.py.py ARN_temp configs/NTU-V2/no-lstm/ARN_temp.cfg NTU-V2 -n 5 -v 2
# ARN_joint+temp
python3 src/train.py.py ARN_joint+temp configs/NTU-V1/no-lstm/ARN_joint+temp.cfg NTU -F middle -n 5 -v 2 
python3 src/train.py.py ARN_joint+temp configs/NTU-V2/no-lstm/ARN_joint+temp.cfg NTU-V2 -F middle -n 5 -v 2

# NTU-V1
# NTU-V1 ARN-LSTM fusion inward+outward
python3 src/train.py.py ARN-LSTM_inward configs/NTU-V1/ARN-LSTM_inward.cfg NTU -n 1 -v 2
python3 src/train.py.py ARN-LSTM_outward configs/NTU-V1/ARN-LSTM_outward.cfg NTU -n 1 -v 2
python3 src/train.py.py ARN-LSTM_inward+outward configs/NTU-V1/ARN-LSTM_inward+outward.cfg NTU -F middle -n 1 -v 2
# NTU-V2
# NTU-V2 ARN-LSTM fusion inward+outward
python3 src/train.py.py ARN-LSTM_inward configs/NTU-V2/ARN-LSTM_inward.cfg NTU-V2 -n 1 -v 2
python3 src/train.py.py ARN-LSTM_outward configs/NTU-V2/ARN-LSTM_outward.cfg NTU-V2 -n 1 -v 2
python3 src/train.py.py ARN-LSTM_inward+outward configs/NTU-V2/ARN-LSTM_inward+outward.cfg NTU-V2 -F middle -n 1 -v 2



