import os

builder = {'-image_source': '', '-template_source': '', '-config_source': '',
                    '-cluster_source': '', '-destination': '', '-num_threads': '',
                    '-write_threshold': '','-validation_dataset': '','-weight_source': '', 
                    '-score_matrixCSV': '', '-edited_photos': ''}

builder['-image_source'] = str('/app/data/images/*')
builder['-template_source'] = str('/app/data/templates/*')
builder['-config_source'] = str('/app/data/config.json')
builder['-cluster_source'] = None
builder['-destination'] = str('/app/data/results')
builder['-num_threads'] = 1
builder['-write_threshold']=30
builder['-validation_dataset'] = str('/app/mask_rcnn/samples/snow_leopard/dataset')
builder['-weight_source'] = str('/app/mask_rcnn/logs/bottle20200221T0110/mask_rcnn_bottle_0010.h5')
builder['-score_matrixCSV'] = str('/app/data/results/score_matrix.csv')
builder['-edited_photos'] = str('/app/data/edited')

command_line = str('python recognition.py')
for argument in builder:
    if builder[argument] != None:
        next = str(' {0} "{1}"'.format(str(argument), str(builder[argument])))
        command_line = command_line + next

os.system(command_line)
