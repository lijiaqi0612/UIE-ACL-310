import argparse
import numpy as np
import struct
import acl
import os
from constant import MemcpyType, MallocType, NpyType


WORK_DIR = os.getcwd()
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0
NPY_FLOAT32 = 11

ret = acl.init()

load_input_dataset = None
load_output_dataset = None
input_data = []
output_data = []
model_desc = 0
run_mode = 0
INDEX = 0
 
# MODEL_PATH = "/root/lijiaqi/UIE/model/h12m0.om"
MODEL_PATH = "/root/lijiaqi/UIE/flask/model/model_uie_v2.om"

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


def _data_interaction_in(model_input_ids, model_token_type_ids, model_position_ids, model_attention_mask):
    global input_data
    temp_data_buffer = input_data
    model_input_ids_np_ptr = acl.util.numpy_to_ptr(model_input_ids)
    model_token_type_ids_np_ptr = acl.util.numpy_to_ptr(model_token_type_ids)
    model_position_ids_np_ptr = acl.util.numpy_to_ptr(model_position_ids)
    model_attention_mask_np_ptr = acl.util.numpy_to_ptr(model_attention_mask)

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

    start_probs = []
    end_probs = []

    for start_prob in dataset[0]:
        start_probs.append(start_prob.tolist())
    
    for end_prob in dataset[1]:
        end_probs.append(end_prob.tolist())
    

    return start_probs, end_probs


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
    