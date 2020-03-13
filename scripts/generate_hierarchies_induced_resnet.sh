for model in ResNet10 ResNet18; do  # add as many as you want!

  for dataset in CIFAR10 CIFAR100 TinyImagenet200; do
    if [ ! -f "checkpoint/ckpt-${dataset}-${model}.pth" ]; then
      python main.py --model=${model} --dataset=${dataset}
    fi

    python generate_hierarchy.py --method=induced --induced-checkpoint=checkpoint/ckpt-${dataset}-${model}.pth --dataset=${dataset}
    python test_generated_graph.py --method=induced --induced-checkpoint=checkpoint/ckpt-${dataset}-${model}.pth --dataset=${dataset}
  done;
done;
