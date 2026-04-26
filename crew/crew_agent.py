import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from crewai import Crew, Agent, Task
from crewai_tools import ScrapeWebsiteTool, FileWriterTool
from llm import qwen_llm


searcher = Agent(
    role="小红书百万爆款文案操盘手",
    goal="每一篇文案都做到点击率拉满、收藏转发暴涨、精准戳中女生情绪、自带爆款体质，轻松上小红书首页推荐流量池",
    backstory="""你深耕小红书爆款内容5年，操盘过美妆、穿搭、生活好物、自律成长、居家日常、母婴、副业等全赛道千万赞笔记。
你深谙小红书算法机制，懂标题黄金公式、开篇留人钩子、情绪共鸣写法、段落短平快、emoji节奏、结尾互动引导、人设氛围感打造。
你写出来的文案自带网感、接地气、不官方、不生硬，句句戳痛点、句句有画面、句句想让人收藏+点赞+评论。
你天然懂得：小红书用户不爱看长文、不爱看广告、爱看真实体验、爱看情绪价值、爱看干货碎碎念、爱看姐妹真心话。
你每一篇笔记都会自动配齐：爆款标题、开头3秒留人、正文分段短句、高频emoji、金句结尾、话题标签，完全对标小红书TOP爆款笔记风格。
你可以通过工具搜索小红书上当前爆款的信息，获取最新最热的爆款内容和趋势。你必须从搜索结果中选择最相关的网页链接，并使用网页抓取工具获取其完整内容。然后，你可以使用这些信息来生成爆款文案。
""",
    # 工具配置：为 Agent 提供完成任务所需的能力
    # ScrapeWebsiteTool：网页抓取工具，用于获取网页的完整内容
    # BaiduSearchTool：百度搜索工具，用于搜索网络信息
    # FileWriterTool：文件写入工具，用于保存调研报告
    tools=[ScrapeWebsiteTool(), FileWriterTool()],
    memory=True,  # 启用记忆功能，Agent 可以记住之前的对话内容
    max_iter=100,  # 最大迭代次数，防止 Agent 陷入无限循环
    llm=qwen_llm.QwenLLM(
        model="qwen-plus",
        api_key=os.getenv("QWEN_API_KEY"),
    ),
)

task = Task(
    description="""根据用户给的产品/主题/内容方向，**全自动生成一篇小红书原生爆款笔记**，严格按照小红书爆款结构写作：

1、标题：必须抓人眼球，带数字/反差/痛点/emoji，一眼就想点进来；
2、开头第一句强钩子，3秒留住人，制造共鸣、焦虑、惊喜、反差感；
3、正文段落全部短句，每段2-3行，密集emoji穿插，阅读零压力；
4、内容风格：姐妹唠嗑语气、真实体验感、不硬广、不官方、网感拉满；
5、内容必须包含：痛点描写→使用感受→真实效果→省钱/避坑/干货心得→走心总结；
6、结尾必须引导评论互动，提升笔记互动率，助力笔记刷上首页流量；
7、最后自动匹配10个小红书高热度精准话题标签。

写作全程不要生硬文案、不要营销感、不要书面化，要像普通女生发自拍日常一样自然又爆火。

用户创作主题：{user_topic}""",
    expected_output="一篇小红书原生爆款风格成品笔记，含爆款标题、正文内容、emoji排版、互动结尾、热门话题标签，可直接复制发布不用修改",
    agent=searcher,  # 指定执行任务的 Agent
)


crew = Crew(
    agents=[searcher],  # 参与工作的 Agent 列表
    tasks=[task],  # 需要执行的任务列表
    verbose=True,  # 启用详细日志，可以看到 Agent 的思考过程
)


# ==============================================================================
# 执行任务
# ==============================================================================
# Crew.kickoff() 启动任务执行流程
# Agent 会按照 ReAct 循环（Reasoning + Acting）来完成任务：
# 1. 思考（Thought）：分析任务，决定下一步行动
# 2. 行动（Action）：调用工具执行操作
# 3. 观察（Observation）：获取工具执行结果
# 4. 重复上述步骤，直到得到最终答案

result = crew.kickoff(inputs={"user_topic": "早八人5分钟伪素颜妆容"})
print(result)