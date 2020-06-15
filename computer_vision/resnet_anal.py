from tensorflow.keras import applications

resnet = applications.ResNet50(weights='imagenet')
resnet.summary()

json_string = resnet.to_json()


# with open('../../data/computer_vision/model.json','w') as f :
#     f.write(json_string)

import json
parse = json.loads(json_string)
# print(json.dumps(parse, indent=4, sort_keys=True))

pp_json_string = json.dumps(parse, indent=4, sort_keys=True)
with open('../../data/computer_vision/model.json','w') as f :
    f.write(pp_json_string)