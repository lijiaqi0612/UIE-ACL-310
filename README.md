# UIE-ACL-310
    
## 需求背景

有一个通用实体关系事件抽取的任务，需要使用到UIE模框架，而且需要将起部署到昇腾310服务器上，因为UIE模型底层使用的是ernie3.0，但是目前paddle官方还不支持ernie3.0模型在昇腾310上部署，所以才有了以下的操作，主要过程是：

1. 先试用paddle训练处模型
2. 然后使用 paddle2onnx.command.c_paddle_to_onnx方法将paddle的模型转为onnx模型 
3. 因现在的onnx模型是动态的shape和散乱的算子形态，需要使用paddle自带的工具paddle2onnx.optimize将onnx模型先进行重塑，固定好shape的维度，将散乱的算子进行整合

    - 命令如下： 
    ```
    $ python -m paddle2onnx.optimize --input_model /home/user/lijiaqi/PaddleNLP/model_zoo/uie/export_new/model.onnx --output_model /home/user/lijiaqi/model_new_uie.onnx --input_shape_dict "{'att_mask':[1,512],'pos_ids':[1,512],'token_type_ids':[1,512],'input_ids':[1,512]}"  
    ```
4. 然后将onnx模型在使用ATC工具转为acl所需要的om模型
5. 另外在使用acl部署的时候，paddle框架是不能使用的，acl使用到的模型和训练过程均需要自己实现，包括from_pretrain阶段的分词，建立词表，数据处理部分，这部分我已经实现完，纯python版本的实现


## 执行过程

模型没有上传，可以按照上面所说的方式训练一版om的模型，或者我后面将om模型上传到百度云，感兴趣的可以自行下载。

直接运行uie_predict.py文件就可以：

```
python uie_predict.py
```


使用flask部署的代码后面可能会更新。
