# Instructions on https://www.tensorflow.org/lite/android/lite_build#configure_workspace_and_bazelrc
docker build . -t tflite-builder -f tflite-android.Dockerfile
MSYS_NO_PATHCONV=1  docker run -it -v $PWD:/host_dir tflite-builder bash 
#-c /host_dir/compile-tflite.sh