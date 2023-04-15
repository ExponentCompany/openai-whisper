#include <jni.h>
#include <string>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstdio>
#include <android/log.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "whisper.h"
#include "input_features.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <algorithm>
//#include "tensorflow/lite/version.h"
#define INFERENCE_ON_AUDIO_FILE 1

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

extern "C" __attribute__ ((visibility ("default"))) int whisper_free_resources( ) {
    return 0;
}

struct MemoryFile {
    char *pointer = nullptr;
    size_t position = 0;
    size_t buffer_size = 0;
};

int MEM_fread(char *buf, size_t size, size_t n, MemoryFile* f) {
    size_t max_n = std::min(size * n, f->buffer_size - f->position) / size;
    memcpy(buf, f->pointer + f->position,  max_n * size);
    f->position += max_n * size;
    return max_n;
}

#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, "Holosweat ASR", __VA_ARGS__)
#if true
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_ERROR, "Holosweat ASR", __VA_ARGS__)
#else
#define LOG_INFO(...)
#endif

#define printf(...) LOG_INFO(__VA_ARGS__)

// Example: load a tflite model using TF Lite C++ API
// Credit to https://github.com/ValYouW/crossplatform-tflite-object-detecion
// Credit to https://github.com/cuongvng/TF-Lite-Cpp-API-for-Android
extern "C" __attribute__ ((visibility ("default"))) int
whisper_run_recognition(
        char* vocab_data,
        size_t vocab_size,
        char* model_data,
        size_t model_size,
        const float* pcm_data,
        int pcm_n_frames,
        char *return_value,
        int max_return_length
        ) {

    //Load Whisper Model into buffer
    return_value[0] = '\0';
     whisper_filters filters;
  whisper_mel mel;
  struct timeval start_time,end_time;
  std::string word;
  int32_t n_vocab = 0;

  MemoryFile vocab_data_stream = {
    .pointer = vocab_data,
    .position = 0,
    .buffer_size = vocab_size
  };

  {
    uint32_t magic=0;
    MEM_fread((char *) &magic, sizeof(magic), 1, &vocab_data_stream);
    //@magic:USEN
    if (magic != 0x5553454e) {
        printf("%s: invalid vocab file (bad magic)\n", __func__);
        return -1;
    }
  }

  // load mel filters
  {
      MEM_fread((char *) &filters.n_mel, sizeof(filters.n_mel), 1, &vocab_data_stream);
      MEM_fread((char *) &filters.n_fft, sizeof(filters.n_fft), 1, &vocab_data_stream);

      filters.data.resize(filters.n_mel * filters.n_fft);
      MEM_fread((char *) filters.data.data(), filters.data.size() * sizeof(float), 1, &vocab_data_stream);
  }

  // load vocab
  {
    MEM_fread((char *) &n_vocab, sizeof(n_vocab), 1, &vocab_data_stream);
    g_vocab.n_vocab = n_vocab;
    printf("\nn_vocab:%d\n",(int)n_vocab);

    for (int i = 0; i < n_vocab; i++) {
      uint32_t len;
      MEM_fread((char *) &len, sizeof(len), 1, &vocab_data_stream);

      word.resize(len);
      MEM_fread((char *) word.data(), len, 1, &vocab_data_stream);
      g_vocab.id_to_token[i] = word;
      //printf("len:%d",(int)len);
      //printf("'%s'\n", g_vocab.id_to_token[i].c_str());
    }

    g_vocab.n_vocab = 51864;//add additional vocab ids
    if (g_vocab.is_multilingual()) {
        g_vocab.token_eot++;
        g_vocab.token_sot++;
        g_vocab.token_prev++;
        g_vocab.token_solm++;
        g_vocab.token_not++;
        g_vocab.token_beg++;
    }
    for (int i = n_vocab; i < g_vocab.n_vocab; i++) {
        if (i > g_vocab.token_beg) {
            word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
        } else if (i == g_vocab.token_eot) {
            word = "[_EOT_]";
        } else if (i == g_vocab.token_sot) {
            word = "[_SOT_]";
        } else if (i == g_vocab.token_prev) {
            word = "[_PREV_]";
        } else if (i == g_vocab.token_not) {
            word = "[_NOT_]";
        } else if (i == g_vocab.token_beg) {
            word = "[_BEG_]";
        } else {
            word = "[_extra_token_" + std::to_string(i) + "]";
        }
        g_vocab.id_to_token[i] = word;
        // printf("%s: g_vocab[%d] = '%s'\n", __func__, i, word.c_str());
    }
  }

  //Generate input_features for Audio file
  std::vector<float> pcmf32;
  {
      // convert to mono, float
      pcmf32.resize(pcm_n_frames);
        for (int i = 0; i < pcm_n_frames; i++) {
            pcmf32[i] = pcm_data[i];
        }

    //Hack if the audio file size is less than 30ms append with 0's
    pcmf32.resize((WHISPER_SAMPLE_RATE*WHISPER_CHUNK_SIZE),0);
    if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 1,filters, mel)) {
      LOG_ERROR("%s: failed to compute mel spectrogram\n", __func__);
      return -1;
    }

    printf("\nmel.n_len%d\n",mel.n_len);
    printf("\nmel.n_mel:%d\n",mel.n_mel);
  }//end of audio file processing

  // Load tflite model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(model_data, model_size);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());
  // Get information about the memory area to use for the model's input.
  float* input = interpreter->typed_input_tensor<float>(0);
  // if (argc == 2) {
  //   memcpy(input, _content_input_features_bin, WHISPER_N_MEL*WHISPER_MEL_LEN*sizeof(float)); //to load pre generated input_features
  // }
  // else if (argc == 3) {
  memcpy(input, mel.data.data(), mel.n_mel*mel.n_len*sizeof(float));
  // }
  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  gettimeofday(&start_time, NULL);
  // Run inference
  if(interpreter->Invoke() != kTfLiteOk) {
      LOG_ERROR("Failed to invoke");
      return -6;
  }
  gettimeofday(&end_time, NULL);
  printf("Inference time %ld seconds \n",(end_time.tv_sec-start_time.tv_sec));
  int output = interpreter->outputs()[0];
  TfLiteTensor *output_tensor = interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  //printf("output size:%d\n",output_size );
  int *output_int = interpreter->typed_output_tensor<int>(0);
  std::string text = "";
  std::string word_add;
  for (int i = 0; i < output_size; i++) {
    //printf("%d\t",output_int[i]);
    if(output_int[i] == g_vocab.token_eot){
      break;
    }
    if (output_int[i] == g_vocab.token_sot || output_int[i] == g_vocab.token_not) {
        continue;
    }
    text += whisper_token_to_str(output_int[i]);
  }
  printf("\n%s\n", text.c_str());
  printf("\n");
    return_value[0] = '\0';
    int return_length = std::min((int)strlen(text.c_str()), max_return_length);
    memcpy(return_value, text.c_str(), return_length);
    return_value[return_length] = '\0';
    return 0;
}