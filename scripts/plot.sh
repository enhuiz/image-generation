mkdir -p figures/$1
python -m src.plot fid --output figures/$1/fid.png ${@:2}
python -m src.plot generator.loss discriminator.loss --output figures/$1/loss.png ${@:2}
