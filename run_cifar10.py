import os
import time
import argparse


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--exclude-classes', type=str, nargs='*')
	parser.add_argument('--exclude-labels', type=str, nargs='*')
	parser.add_argument('--method', type=str, default='induced')
	parser.add_argument('--num-samples',type=str, default='5')
	return parser
	
if __name__=='__main__':
	args = get_parser().parse_args()
	cmds = []

	#print('step 1 : train model. cmd to be submitted... in 5 second')
	arg_exclude_classes = ' '.join(args.exclude_classes)
	arg_checkpoint_fname = '-'.join(['ckpt',args.model,'CIFAR10','exclude',*args.exclude_classes])
	cmd = ' '.join(['python', 'main.py', '--model', args.model,'--dataset','CIFAR10ExcludeClasses', '--exclude-classes', arg_exclude_classes,'--checkpoint-fname', arg_checkpoint_fname])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	#print('step 2 : generate hierarchy. cmd to be sumitted ... in 5 second')
	arg_exclude_labels = ' '.join(args.exclude_labels)
	arg_induced_checkpoint = './checkpoint/' + arg_checkpoint_fname + '.pth'
	cmd = ' '.join(['python', 'generate_hierarchy.py', '--method', args.method, '--dataset', 'CIFAR10', '--exclude-labels', arg_exclude_labels,
			'--induced-checkpoint', arg_induced_checkpoint])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	#print('step 3 : fine tune. cmd to be submitted in 5 sec')
	arg_path_graph = './data/CIFAR10/' + '-'.join(['graph','induced','exclude',*args.exclude_classes]) + '.json'
	cmd = ' '.join(['python','main.py','--model',args.model,'--dataset','CIFAR10ExcludeClasses','--exclude-classes',arg_exclude_classes,'--resume','--path-resume',arg_induced_checkpoint,'--loss','SoftTreeSupLoss','--path-graph',arg_path_graph, '--freeze-conv','--analysis', 'SoftEmbeddedDecisionRules'])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	#print('step 4 : add weight vector. cmd to be sub in 5 sec')
	arg_new_checkpoint_fname = '-'.join(['ckpt',args.model,'CIFAR10','exclude',*args.exclude_classes,'weighted'])
	cmd = ' '.join(['python', 'add_zeroshot_vec.py', '--model', args.model, '--resume', '--path-resume', arg_induced_checkpoint, '--new-classes', arg_exclude_classes, '--num-samples', args.num_samples, '--checkpoint-fname', arg_new_checkpoint_fname])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	#print('step 5 : reform hierarchy, in 5 sec')
	arg_new_induced_checkpoint = './checkpoint/' + arg_new_checkpoint_fname + '.pth'
	cmd = ' '.join(['python', 'generate_hierarchy.py', '--method', args.method, '--dataset', 'CIFAR10', '--ignore-labels', arg_exclude_labels, '--induced-checkpoint', arg_new_induced_checkpoint])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	#print('step 6 : evaluate, in 5 sec')
	arg_new_path_graph = './data/CIFAR10/' + '-'.join(['graph','induced','exclude',*args.exclude_classes,'weighted']) + '.json'
	cmd = ' '.join(['python','main.py','--model', args.model,'--dataset', 'CIFAR10', '--resume', '--eval', '--path-resume', arg_new_induced_checkpoint, '--path-graph', arg_new_path_graph, '--analysis', 'SoftEmbeddedDecisionRules'])
	#print(cmd)
	#time.sleep(5)
	#os.system(cmd)
	cmds.append(cmd)

	print()
	for cmd in cmds: print(cmd)
	option = input('Submit the above commands? y/n ')
	if option=='y':
		for cmd in cmds: os.system(cmd)
	else:
		pass


"""
python main.py --model ResNet12 --dataset CIFAR10 --resume --eval --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat-weighted.pth --path-graph [JSON FILE OF HIERARCHY] --analysis SoftEmbeddedDecisionRules
 
python generate_hierarchy.py --method induced --dataset CIFAR10 --ignore-labels 3 --induced-checkpoint ./checkpoint/ckpt-CIFAR10-exclude-cat-weighted.pth

python add_zeroshot_vec.py --model ResNet12 --resume --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat.pth --new-classes cat --num-samples 5 --checkpoint-fname ckpt-CIFAR10-exclude-cat-weighted

python main.py --model ResNet12 --dataset CIFAR10ExcludeClasses --exclude-classes cat --resume --path-resume ./checkpoint/ckpt-CIFAR10-exclude-cat.pth --loss SoftTreeSupLoss --path-graph [JSON FILE OF HIERARCHY] --freeze-conv --analysis SoftEmbeddedDecisionRules
	
python generate_hierarchy.py --method induced --dataset CIFAR10 --exclude-labels 3 --induced-checkpoint ./checkpoint/ckpt-CIFAR10-exclude-cat.pth

python main.py --model ResNet12 --dataset CIFAR10ExcludeClasses --exclude-classes cat --checkpoint-fname ckpt-CIFAR10-exclude-cat

"""
