from vocab import Vocabulary
import evaluation
evaluation.evalrank("/home/lihaoxuan/liaojunrong/cross-modal-retrieval/SCAN-master/runs/runX/checkpoint/model_best.pth.tar",
                    data_path="/data1/lihaoxuan/LJR_data", split="test")