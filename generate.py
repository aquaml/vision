import pipelines
import subprocess
import csv
import aqua
import random
import time
import argparse
import torch

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b', '--batch_size', type=int, required=True, help='an integer for batch size')
parser.add_argument('-d', '--duration', type=int, required=True, help='duration of the experiment in minutes')
parser.add_argument('-m', '--model', choices=['sdxl', 'sd', 'kand'], required=True, help='model name')

args = parser.parse_args()

batch_size = args.batch_size
duration_mins = args.duration

if args.model == 'sdxl':
    pipeline_type = pipelines.Pipeline.SDXL_PIPELINE
elif args.model == 'sd':
    pipeline_type = pipelines.Pipeline.SD_PIPELINE
elif args.model == 'kand':
    pipeline_type = pipelines.Pipeline.KANDINSKY
else:
    raise Exception("invalid pipeline")

mmc = aqua.static.static_informer('localhost', 8080, 0)
offered_memory = False

def current_time_millis():
    return round(time.time() * 1000)

def parse_prompts():
    prompt_list = []
    with open('prompts.tsv', 'r') as prompt_file:
        reader = csv.reader(prompt_file, delimiter='\t')
        for row in reader:
            break
        for row in reader:
            prompt = row[0]
            prompt_list.append(prompt)
    return prompt_list

def get_prompts(prompts, k):
    return random.sample(prompts, k)

#sd 20, sdxl 8, kand 20
def get_gpu_memory():
    cmd = "nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output_str = result.stdout.decode('utf-8')
    output_str = output_str.split('\n')[2]
    parts = output_str.split(',')
    return parts

def torch_memory():
    global offered_memory
    if not offered_memory:
        free_memory, _ = torch.cuda.mem_get_info()
        offered_memory = True
        mmc.offer_memory(free_memory)
    
def get_gpu_memory_str():
    labels = ['used=','total=','free=']
    parts = get_gpu_memory()
    for idx, part in enumerate(parts):
        parts[idx] = '{}{}'.format(labels[idx], part.strip())
    return ','.join(parts)


pipeline, generator = pipelines.get_pipeline(pipeline_type, 31)
pipeline = pipeline.to('cuda')
prompt_list = parse_prompts()

start_time = current_time_millis()
duration_of_loop = 60 * 1000 * duration_mins # milliseconds
one_minute_in_ms = 60 * 1000
num_generated = 0

logging_num_generated = 0
prev_logged_time = start_time

while current_time_millis() - start_time < duration_of_loop:
    prompt = get_prompts(prompt_list, batch_size)
    print(prompt)
    images = pipeline(prompt, generator=generator).images
    # for image_id, image in enumerate(images):
    #     image.save('img_xl_{}.png'.format(num_generated + image_id))
    print(get_gpu_memory_str())
    torch_memory()
    num_generated += batch_size
    time_elapsed = (current_time_millis() - prev_logged_time) / 1000
    logging_num_generated += batch_size

    if time_elapsed >= 10:
        throughput = logging_num_generated / ((current_time_millis() * 1.0 - start_time * 1.0) / one_minute_in_ms)
        print('Throughput: {} images/minute'.format(throughput))
        logging_num_generated = 0


end_time = current_time_millis()

throughput = num_generated / ((end_time * 1.0 - start_time * 1.0) / one_minute_in_ms)
print("Throughput is: {} images/min".format(throughput))

