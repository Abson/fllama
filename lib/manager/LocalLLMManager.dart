import 'dart:async';

import 'package:fllama/fllama.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tiktoken/flutter_tiktoken.dart';
import 'package:flutter/foundation.dart';
import 'LLMException.dart';
import 'SummaryExtractor.dart';
import 'TextTokenSplitter.dart';

/// 本地LLM响应监听器接口
typedef LLMResponseListener = void Function(String content, bool done);

/// 本地LLM管理器，用于处理模型推理请求
class LocalLLMManager {
  int? _runningRequestId;

  static final LocalLLMManager _instance = LocalLLMManager._();

  static LocalLLMManager get instance => _instance;

  LocalLLMManager._();

  /// 运行推理并获取响应
  /// [modelPath] 模型文件路径
  /// [content] 输入内容
  /// [listener] 响应监听器
  /// [maxTokens] 最大生成Token数
  /// [temperature] 温度参数
  /// [topP] 采样参数
  /// [contextSize] 上下文窗口大小
  Future<String> runInference({
    required String modelPath,
    required String content,
    required LLMResponseListener listener,
    int maxTokens = 1000,
    double temperature = 0.0,
    // double topP = 0.9,
    int contextSize = 2048,
    void Function(String log)? logge,
  }) async {
    cancelInference();
    logge?.call("run model modelPath:$modelPath content_len:${content.length}");
    String allResult = "";
    try {
      logge?.call("total tokens:${calcuateTokens(text: content)}");
      // 文章切片
      List<String> textChunks = TextTokenSplitter.splitTextByTokens(
        text: content,
        maxTokensPerChunk: 1536,
        modelName: 'gpt-4', // 可选，默认就是gpt-4
        minLastChunkSize: 200,
      );

      logge?.call('text has been split into ${textChunks.length} 段');
      for (int i = 0; i < textChunks.length; i++) {
        logge?.call(
          '${i + 1} chunks: ${TextTokenSplitter.calcuateTokens(text: textChunks[i])} tokens',
        );
      }
      String chunk = "";
      int totalToken = 0;
      String latestResultString = "";
      for (int i = 0; i < textChunks.length; i++) {
        chunk = textChunks[i];
        totalToken = totalToken + calcuateTokens(text: chunk);
        if (totalToken > 6000) {}
        final completer = Completer();
        // 2. Inference with chat template.
        final request = OpenAiRequest(
          maxTokens: maxTokens.round(),
          messages: [
            Message(Role.user, """
Please summarize the following transcription content in 30-90 words that captures the core theme. Return ONLY the summary itself under the heading "## Summary" without any explanations, introductions, or additional text.

Transcription content:
[$chunk]

This will give you just the requested format:

## Summary
xxxxxx            
            """)
          ],
          numGpuLayers: 99,
          /* this seems to have no adverse effects in environments w/o GPU support, ex. Android and web */
          modelPath: modelPath,
          // mmprojPath: _mmprojPath,
          frequencyPenalty: 0.0,
          // Don't use below 1.1, LLMs without a repeat penalty
          // will repeat the same token.
          presencePenalty: 1.1,
          topP: 1.0,
          // contextSize: 20000,
          // Don't use 0.0, some models will repeat
          // the same token.
          temperature: temperature,
          contextSize: contextSize,
          logger: (log) {
            if (log.contains('ggml_')) {
              return;
            }
            // ignore: avoid_print
            if (kDebugMode) debugPrint('[llama.cpp] $log');
            logge?.call('[llama.cpp] $log');
          },
        );

        int requestId = await fllamaChat(
          request,
          (response, responseJson, done) {
            if (kDebugMode) {
              debugPrint(
                  "[$runtimeType] done:$done response string length:${response.length}");
            }
            if (response.startsWith("Error:")) {
              if (kDebugMode) {
                debugPrint("[$runtimeType] LLM callback error $response");
              }
              completer.completeError(LLMException(response, -1));
            }
            if (response.contains("<end_of_turn>")) {
              done = true;
            }
            if (done) {
              allResult = "$allResult\n\n$latestResultString";
              completer.complete();
            } else {
              latestResultString = response;
              if (response.isNotEmpty) {
                listener(allResult + response, false);
              }
            }
          },
        );
        _runningRequestId = requestId;
        // 等待当前分片处理完成
        await completer.future;
        cancelInference();
      }
    } catch (e) {
      logge?.call("run inference error $e");
      rethrow;
    }
    logge?.call("finish llm");
    logge?.call('allResult $allResult');
    List<String> summaries =
        SummaryExtractor.extractSummaries(allResult); // 打印结果
    StringBuffer result = StringBuffer();
    for (int i = 0; i < summaries.length; i++) {
      if (summaries[i].isNotEmpty) {
        result.writeln('- ${summaries[i]}');
      }
    }

    result.writeln(allResult);
    String summariesResult = result.toString().replaceAll(RegExp(r'```'), "");
    listener(summariesResult, true);
    if (calcuateTokens(text: summariesResult) > 2100) {
      return content;
    }
    return trySecondarySummary(
        allResult
            // 避免 AI 重复总结
            .replaceAll(RegExp(r'## summary'), "")
            .replaceAll(RegExp(r'## Summary'), ""),
        modelPath);
  }

  /**
   * 第二次总结
   */
  Future<String> trySecondarySummary(content, modelPath) async {
    final completer = Completer();
    String summaryResult = "";
    final request = OpenAiRequest(
      maxTokens: 2048,
      messages: [
        Message(Role.user, """
# Text Summary Prompt

Please provide a comprehensive summary of the provided text, structured in the following format:

```
# Summary

**Topic**: [Summarize the core subject of the text in 1-2 sentences]

**Key Points**:
- [List first key point]
- [List second key point]
- [Continue listing all important key points, ensuring each is concise]
- [Focus on significant events, turning points, challenges, and achievements]

**Conclusion**:
- [Provide 1-2 paragraphs of concluding thoughts on the overall content, emphasizing core lessons or insights]
```

When analyzing the text, please:
1. Identify and distill the most important information
2. Arrange key points in chronological order or by importance
3. Reflect deeper meanings or lessons in the conclusion
4. Remain objective and ensure the summary accurately represents the original content

# Content
[$content]
            """)
//         Message(Role.user, """
// Please provide a brief summary of the given text, using this streamlined format:
//
// ```
// # Summary
//
// **Topic**: [Summarize the core subject in one sentence]
//
// **Key Points**:
// - [First key point, maximum 30 words]
// - [Second key point, maximum 30 words]
// - [List up to 5 most important points only]
//
// **Conclusion**: [Summarize the core insights in one brief paragraph (maximum 60 words)]
// ```
//
// Summary requirements:
// 1. Extract only the most essential, valuable information
// 2. Ensure brevity and precision, avoid redundancy
// 3. Remain objective and accurately reflect the original content
//
//
// # Content
// [$content]
//             """)
      ],
      numGpuLayers: 99,
      /* this seems to have no adverse effects in environments w/o GPU support, ex. Android and web */
      modelPath: modelPath,
      // mmprojPath: _mmprojPath,
      frequencyPenalty: 0.0,
      // Don't use below 1.1, LLMs without a repeat penalty
      // will repeat the same token.
      presencePenalty: 1.1,
      topP: 1.0,
      // contextSize: 20000,
      // Don't use 0.0, some models will repeat
      // the same token.
      temperature: 0.0,
      contextSize: 2048,
      logger: (log) {
        if (log.contains('ggml_')) {
          return;
        }
        // ignore: avoid_print
        if (kDebugMode) debugPrint('[llama.cpp] $log');
      },
    );

    int requestId = await fllamaChat(
      request,
      (response, responseJson, done) {
        if (kDebugMode) {
          debugPrint(
              "[$runtimeType] done:$done response string length:${response.length}");
        }
        print(
            "[$runtimeType] done:$done response string length:${response.length}");
        if (response.startsWith("Error:")) {
          if (kDebugMode) {
            debugPrint("[$runtimeType] LLM callback error $response");
          }
          completer.completeError(LLMException(response, -1));
        }
        if (response.contains("<end_of_turn>")) {
          done = true;
        }
        if (done) {
          completer.complete();
        } else {
          summaryResult = response;
          print("赋值 $response\n$summaryResult");
        }
      },
    );

    _runningRequestId = requestId;
    // 等待当前分片处理完成
    await completer.future;
    print("secondary summary result:$summaryResult");
    return summaryResult;
  }

  /// 取消当前正在运行的推理请求
  void cancelInference() {
    if (_runningRequestId != null) {
      fllamaCancelInference(_runningRequestId!);
      _runningRequestId = null;
    }
  }

  static int calcuateTokens({
    required String text,
    String modelName = 'gpt-4',
  }) {
    final encoding = encodingForModel(modelName);
    final numTokens = encoding.encode(text).length;
    print("calcuateTokens result:$numTokens");
    return numTokens;
  }
}
