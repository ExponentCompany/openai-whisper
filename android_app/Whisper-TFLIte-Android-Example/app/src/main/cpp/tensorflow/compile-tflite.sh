VERSION=2.13.0
wget "https://github.com/tensorflow/tensorflow/archive/refs/tags/v${VERSION}.tar.gz" -O /tmp/tf.tar.gz
tar -xvf /tmp/tf.tar.gz
cd tensorflow-$VERSION
cat > .tf_configure.bazelrc <<EOF
build --action_env ANDROID_NDK_HOME="$ANDROID_NDK_HOME"
build --action_env ANDROID_NDK_API_LEVEL="$ANDROID_NDK_API_LEVEL"
build --action_env ANDROID_BUILD_TOOLS_VERSION="$ANDROID_BUILD_TOOLS_VERSION"
build --action_env ANDROID_SDK_API_LEVEL="$ANDROID_API_LEVEL"
build --action_env ANDROID_SDK_HOME="$ANDROID_SDK_HOME"
EOF

sdkmanager \
  "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
  "platform-tools" \
  "platforms;android-${ANDROID_API_LEVEL}"

bazel build -c opt --fat_apk_cpu=x86_64,arm64-v8a   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain   --define=android_dexmerger_tool=d8_dexmerger   --define=android_incremental_dexing_tool=d8_dexbuilder   //tensorflow/lite/c:tensorflowlite_c
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so /host_dir/

# Directory where files will be copied
HOST_DIR="/host_dir/include"

# Execute the Bazel query and process its output
bazel query 'kind("source file", deps(//tensorflow/lite/c:tensorflowlite_c))' --output=location | grep '\.h$' | while read -r line; do
    # Extract the file path using cut
    filepath=$(echo "$line" | cut -d ':' -f 1)

    if [[ $line == *"/external/flatbuffers"* ]]; then
        # Extract relative path after external
        relpath=${filepath#*"/external/flatbuffers"}
        dest="$HOST_DIR/external/flatbuffers/$relpath"
    elif [[ $line == *"/tensorflow/lite"* ]]; then
        # Extract relative path after external
        relpath=${filepath#*"/tensorflow/lite"}
        dest="$HOST_DIR/tensorflow/lite/$relpath"
    else
        continue
    fi

    # Create directory structure
    mkdir -p "$(dirname "$dest")"
    
    # Copy the file
    cp "$filepath" "$dest"
    
done