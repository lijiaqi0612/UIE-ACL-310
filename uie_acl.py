import argparse
import numpy as np
import struct
import acl
import os
from constant import MemcpyType, MallocType, NpyType
# from util import *


WORK_DIR = os.getcwd()
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0
NPY_FLOAT32 = 11

ret = acl.init()

# GLOBAL
load_input_dataset = None
load_output_dataset = None
input_data = []
output_data = []
_output_info = []
# images_list = []
model_desc = 0
run_mode = 0
INDEX = 0

# MODEL_PATH = "/root/lijiaqi/UIE/model/h12m0.om"
MODEL_PATH = "/root/lijiaqi/UIE/model/model_tiny_out.om"

# if WORK_DIR.find("src") == -1:
#     MODEL_PATH = WORK_DIR + "/src/model/googlenet_yuv.om"
#     DATA_PATH = WORK_DIR + "/src/data"
# else:
#     MODEL_PATH = WORK_DIR + "/model/googlenet_yuv.om"
#     DATA_PATH = WORK_DIR + "/data"

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }

def check_ret(message, ret):
    if ret != ACL_ERROR_NONE:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

    
def init():
    ret = acl.init()
    check_ret("acl.init", ret)
    # print("init success")

    
def allocate_res(device_id):   
    ret = acl.rt.set_device(device_id)
    check_ret("acl.rt.set_device", ret)
    context, ret = acl.rt.create_context(device_id)
    check_ret("acl.rt.create_context", ret)
    # print("allocate_res success")
    return context


def load_model(model_path):
    model_id, ret = acl.mdl.load_from_file(model_path)
    check_ret("acl.mdl.load_from_file", ret)
    # print("load_model success")
    return model_id

def get_model_data(model_id):
    global model_desc
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    check_ret("acl.mdl.get_desc", ret)

    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    # print("get_model_data success")
    return input_size, output_size

def gen_data_buffer(num, des):
    global model_desc
    func = buffer_method[des]
    for i in range(num):
        #temp_buffer_size = (model_desc, i)
        # temp_buffer_size  = acl.mdl.get_output_size_by_index(model_desc, i)
        # temp_buffer, ret = acl.rt.malloc(temp_buffer_size, MallocType.ACL_MEM_MALLOC_NORMAL_ONLY.value)
        # check_ret("acl.rt.malloc", ret)
        if des == "in":
            temp_buffer_size  = acl.mdl.get_input_size_by_index(model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, MallocType.ACL_MEM_MALLOC_HUGE_FIRST.value)
            check_ret("acl.rt.malloc", ret)
            input_data.append({"buffer": temp_buffer,
                                    "size": temp_buffer_size})
        elif des == "out":
            temp_buffer_size  = acl.mdl.get_output_size_by_index(model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, MallocType.ACL_MEM_MALLOC_NORMAL_ONLY.value)
            check_ret("acl.rt.malloc", ret)
            output_data.append({"buffer": temp_buffer,
                                     "size": temp_buffer_size})


def malloc_device(input_num, output_num):
    gen_data_buffer(input_num, des="in")
    gen_data_buffer(output_num, des="out")


# def _data_interaction_in(dataset):
#     global input_data
#     temp_data_buffer = input_data
#     for i in range(len(temp_data_buffer)):
#         item = temp_data_buffer[i]
#         ptr = acl.util.numpy_to_ptr(dataset)
#         ret = acl.rt.memcpy(item["buffer"],
#                             item["size"],
#                             ptr,
#                             item["size"],
#                             ACL_MEMCPY_HOST_TO_DEVICE)
#         check_ret("acl.rt.memcpy", ret)
#     print("data_interaction_in success")

def _data_interaction_in(model_input_ids, model_token_type_ids, model_position_ids, model_attention_mask):
    global input_data
    temp_data_buffer = input_data
    model_input_ids_np_ptr = acl.util.numpy_to_ptr(model_input_ids)
    model_token_type_ids_np_ptr = acl.util.numpy_to_ptr(model_token_type_ids)
    model_position_ids_np_ptr = acl.util.numpy_to_ptr(model_position_ids)
    model_attention_mask_np_ptr = acl.util.numpy_to_ptr(model_attention_mask)

    # print(model_input_ids)
    # print(model_token_type_ids)
    # print(model_position_ids)
    # print(model_attention_mask)
    # exit()

    ret = acl.rt.memcpy(temp_data_buffer[0]["buffer"], temp_data_buffer[0]["size"], model_attention_mask_np_ptr, temp_data_buffer[0]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
    ret1 = acl.rt.memcpy(temp_data_buffer[1]["buffer"], temp_data_buffer[1]["size"], model_position_ids_np_ptr, temp_data_buffer[1]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
    ret2 = acl.rt.memcpy(temp_data_buffer[2]["buffer"], temp_data_buffer[2]["size"], model_token_type_ids_np_ptr, temp_data_buffer[2]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
    ret3 = acl.rt.memcpy(temp_data_buffer[3]["buffer"], temp_data_buffer[3]["size"], model_input_ids_np_ptr, temp_data_buffer[3]["size"], ACL_MEMCPY_HOST_TO_DEVICE)

    check_ret("acl.rt.memcpy", ret)
    check_ret("acl.rt.memcpy", ret1)
    check_ret("acl.rt.memcpy", ret2)
    check_ret("acl.rt.memcpy", ret3)

    # print("data_interaction_in success")

def create_buffer(dataset, type="in"):
    global input_data, output_data
    if type == "in":    
        temp_dataset = input_data
    else:
        temp_dataset = output_data
    for i in range(len(temp_dataset)):
        item = temp_dataset[i]
        data = acl.create_data_buffer(item["buffer"], item["size"])
        if data is None:
            ret = acl.destroy_data_buffer(dataset)
            check_ret("acl.destroy_data_buffer", ret)
        _, ret = acl.mdl.add_dataset_buffer(dataset, data)
        if ret != ACL_ERROR_NONE:
            ret = acl.destroy_data_buffer(dataset)
            check_ret("acl.destroy_data_buffer", ret)
    # print("create data_buffer {} success".format(type))

def _gen_dataset(type="in"):
    global load_input_dataset, load_output_dataset
    dataset = acl.mdl.create_dataset()
    #print("create data_set {} success".format(type))
    if type == "in":    
        load_input_dataset = dataset
    else:
        load_output_dataset = dataset
    create_buffer(dataset, type)

def inference(model_id, _input, _output):
    global load_input_dataset, load_output_dataset
    ret = acl.mdl.execute(model_id,
                    load_input_dataset,
                    load_output_dataset)
    check_ret("acl.mdl.execute", ret)
   

def _destroy_data_set_buffer():
    global load_input_dataset, load_output_dataset
    for dataset in [load_input_dataset, load_output_dataset]:
        if not dataset:
            continue
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        ret = acl.mdl.destroy_dataset(dataset)
        check_ret("acl.mdl.destroy_dataset", ret)

def _data_interaction_out(dataset):
    global output_data
    temp_data_buffer = output_data
    if len(dataset) == 0:
        for item in output_data:
            temp, ret = acl.rt.malloc_host(item["size"])
            if ret != 0:
                raise Exception("can't malloc_host ret={}".format(ret))
            dataset.append({"size": item["size"], "buffer": temp})
    for i in range(len(temp_data_buffer)):
        item = temp_data_buffer[i]
        ptr = dataset[i]["buffer"]
        ret = acl.rt.memcpy(ptr,
                            item["size"],
                            item["buffer"],
                            item["size"],
                            ACL_MEMCPY_DEVICE_TO_HOST)
        check_ret("acl.rt.memcpy", ret)
    
    # print(dataset)

def print_result(result):
    dataset = []
    for i in range(len(result)):
        temp = result[i]
        # size = temp["size"]/8
        ptr = temp["buffer"]
        data = acl.util.ptr_to_numpy(ptr, (1, 512), NPY_FLOAT32)
        dataset.append(data)
    
    # print(dataset)
    # print(len(dataset))
    # print(len(dataset[0]))
    # print(dataset[0], type(dataset[0]))
    # print(dataset[1])
    # print(dataset[0][0][:20])
    # print(dataset[0][1][:20])
    # exit()

    start_probs = []
    end_probs = []

    for start_prob in dataset[0]:
        start_probs.append(start_prob.tolist())
    
    for end_prob in dataset[1]:
        end_probs.append(end_prob.tolist())
    

    return start_probs, end_probs

            
    # # predictorV2 = UIEPredictor(512, 2, schema1, position_prob=0.5) 
    # temp = predictor.predict(texts, 'out', start_probs=start_probs, end_probs=end_probs)

    # print(temp)
    # st = struct.unpack("1000f", bytearray(dataset[0]))
    # vals = np.array(st).flatten()
    # top_k = vals.argsort()[-1:-6:-1]
    # print()
    # print("======== image: {} =============".format(images_list[INDEX]))
    # print("======== top5 inference results: =============")
    # INDEX+=1
    # for n in top_k:
    #     object_class = get_image_net_class(n)
    #     print("label:%d  confidence: %f, class: %s" % (n, vals[n], object_class))

def release(model_id, context):
    global input_data, output_data
    ret = acl.mdl.unload(model_id)
    check_ret("acl.mdl.unload", ret)
    while input_data:
        item = input_data.pop()
        ret = acl.rt.free(item["buffer"])
        check_ret("acl.rt.free", ret)
    while output_data:
        item = output_data.pop()
        ret = acl.rt.free(item["buffer"])
        check_ret("acl.rt.free", ret)
    if context:
        ret = acl.rt.destroy_context(context)
        check_ret("acl.rt.destroy_context", ret)
        context = None
    ret = acl.rt.reset_device(0)
    check_ret("acl.rt.reset_device", ret)
    # print('release source success')
    


# def main():
#     global input_data 
#     #init()
#     context = allocate_res(0)
#     model_id = load_model(MODEL_PATH) 
#     print("model_id:{}".format(model_id))
#     input_num, output_num = get_model_data(model_id)
#     malloc_device(input_num, output_num) 
#     # dvpp = Dvpp()
#     # img_list = image_process_dvpp(dvpp)
#     device = 0

#     # texts = [
#     #     '"北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"',
#     #     '原告赵六，2022年5月29日生\n委托代理人孙七，深圳市C律师事务所律师。\n被告周八，1990年7月28日出生\n委托代理人吴九，山东D律师事务所律师'
#     # ]
#     # schema1 = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
#     # schema2 = [{'原告': ['出生日期', '委托代理人']}, {'被告': ['出生日期', '委托代理人']}]
#     texts = ['2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！', '']
#     schema1 = ['时间', '选手', '赛事名称']
#     # schema1 = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
#     result_type = 'in'
#     predictor = UIEPredictor(512, 2, schema1, position_prob=0.5)
#     print("-----------------------------")
#     outputs = predictor.predict(texts, result_type)

#     model_input_ids_list = outputs['input_ids'].astype(np.int64)
#     model_token_type_ids_list = outputs['token_type_ids'].astype(np.int64)
#     model_position_ids_list = outputs['pos_ids'].astype(np.int64)
#     model_attention_mask_list = outputs['att_mask'].astype(np.int64)

    
#     # for i in range(len(model_input_ids_list)):
#     #     sample.forward(model_input_ids_list[i], model_token_type_ids_list[i], model_position_ids_list[i], model_attention_mask_list[i])




#     # for i in range(len(model_input_ids_list)):
#     #     # image_data = {"buffer":image.data(), "size":image.size}   
        
#     #     _data_interaction_in(model_input_ids_list[i], model_token_type_ids_list[i], model_position_ids_list[i], model_attention_mask_list[i])
#     #     _gen_dataset("in")
#     #     _gen_dataset("out")
#     #     inference(model_id, load_input_dataset, load_output_dataset)
#     #     _destroy_data_set_buffer()
#     #     res = []
#     #     _data_interaction_out(res)
#     #     print_result(res)
#     _data_interaction_in(model_input_ids_list, model_token_type_ids_list, model_position_ids_list, model_attention_mask_list)
#     _gen_dataset("in")
#     _gen_dataset("out")
#     inference(model_id, load_input_dataset, load_output_dataset)
#     _destroy_data_set_buffer()
#     res = []
#     _data_interaction_out(res)
#     print_result(res, texts, schema1, predictor)
#     release(model_id,context)

# if __name__ == '__main__':
#     main()

