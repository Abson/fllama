class SummaryExtractor {
  /// 解析文本，提取所有Summary部分
  ///
  /// 接收一个文本字符串，返回所有"## Summary"部分的内容列表
  /// 例如：extractSummaries(text) => ["第一个summary内容", "第二个summary内容",...]
  static List<String> extractSummaries(String text) {
    List<String> summaries = [];

    // 使用正则表达式匹配## Summary及其后面的内容，直到下一个##开头的部分
    // RegExp regExp = RegExp(r'## Summary\s+(.*?)(?=\s*##|\s*$)',
    //     dotAll: true); // dotAll允许.匹配换行符// 找出所有匹配项
    final RegExp regExp = RegExp(
      r'## Summary(?!\s+Brief)\s+\n*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*\n|\s*##|\s*$)',
      dotAll: true,
    );
    Iterable<Match> matches = regExp.allMatches(text);

    // 将每个匹配项添加到结果列表中
    for (var match in matches) {
      if (match.group(1) != null) {
        String summary = match.group(1)!.trim();
        summaries.add(summary);
      }
    }
    return summaries;
  }
}
