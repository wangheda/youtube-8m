
num=0
for filename in $(ls /Youtube-8M/video/test); do
	if [[ $filename =~ ^test ]]; then
		num=$(( $num + 1 ))
		echo "echo 'processing test file no.$num' && (CUDA_VISIBLE_DEVICES='' python filetransformer.py /Youtube-8M/video/test/$filename mean_rgb 1024 | gzip -c > ~/Youtube-8M/video/test/${filename}.vec.gz)"
	fi
done

num=0
for filename in $(ls /Youtube-8M/video/train); do
	if [[ $filename =~ ^train ]]; then
		num=$(( $num + 1 ))
		echo "echo 'processing train file no.$num' && (CUDA_VISIBLE_DEVICES='' python filetransformer.py /Youtube-8M/video/train/$filename mean_rgb 1024 | gzip -c > ~/Youtube-8M/video/train/${filename}.vec.gz)"
	fi
done

num=0
for filename in $(ls /Youtube-8M/video/validate); do
	if [[ $filename =~ ^validate ]]; then
		num=$(( $num + 1 ))
		echo "echo 'processing validate file no.$num' && (CUDA_VISIBLE_DEVICES='' python filetransformer.py /Youtube-8M/video/validate/$filename mean_rgb 1024 | gzip -c > ~/Youtube-8M/video/validate/${filename}.vec.gz)"
	fi
done

