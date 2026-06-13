import type { Locale } from "./i18n";

export interface Pillar {
  icon: string;
  title: string;
  body: string;
}

export interface Dictionary {
  nav: { tseval: string; research: string; news: string; join: string };
  themeToggle: { toLight: string; toDark: string };
  hero: {
    eyebrow: string;
    title: string;
    titleAccent: string;
    lede: string;
    ctaNews: string;
    ctaGithub: string;
    etymology: string;
  };
  mission: {
    label: string;
    heading: string;
    body: string;
  };
  pillars: {
    label: string;
    heading: string;
    items: Pillar[];
  };
  latest: {
    researchLabel: string;
    newsLabel: string;
    viewAll: string;
    empty: string;
  };
  cta: {
    heading: string;
    body: string;
  };
  research: { title: string; lede: string };
  news: { title: string; lede: string };
  /** TS-Eval leaderboard page. */
  tseval: {
    title: string;
    lede: string;
    updated: string;
    submissions: string;
    rankedBy: string;
    horizon: string;
    emptyTrack: string;
    tracks: Record<string, string>;
    cols: { model: string; mse: string; mae: string; runs: string };
    categories: {
      commonStatic: string;
      commonStaticZh: string;
      realtime: string;
      realtimeZh: string;
    };
    filters: {
      title: string;
      resetAll: string;
      searchPlaceholder: string;
      datasetCategories: string;
      display: string;
      metrics: string;
      options: string;
    };
    datasets: {
      etth1: string;
      etth2: string;
      ettm1: string;
      ettm2: string;
      traffic: string;
      solar: string;
      electricity: string;
      weather: string;
    };
    datasetDescriptions: {
      hourly1: string;
      hourly2: string;
      minute1: string;
      minute2: string;
      roadTraffic: string;
      solarPower: string;
      powerUsage: string;
      weatherData: string;
    };
    displayOptions: {
      showMedals: string;
      showMedalsDesc: string;
      performanceBars: string;
      performanceBarsDesc: string;
      highlightTop3: string;
      highlightTop3Desc: string;
      compactMode: string;
      compactModeDesc: string;
      submissionIds: string;
      submissionIdsDesc: string;
      colorByRank: string;
      colorByRankDesc: string;
    };
    metricsOptions: {
      mse: string;
      mseDesc: string;
      mae: string;
      maeDesc: string;
      rmse: string;
      rmseDesc: string;
      corr: string;
      corrDesc: string;
      fitTime: string;
      fitTimeDesc: string;
      inferenceTime: string;
      inferenceTimeDesc: string;
      runs: string;
      runsDesc: string;
    };
    modelTypes: {
      title: string;
      timeSeries: string;
      timeSeriesDesc: string;
      spatialTemporal: string;
      spatialTemporalDesc: string;
    };
    viewModes: {
      table: string;
      compact: string;
      detailed: string;
    };
    navigation: {
      previous: string;
      next: string;
      dataset: string;
      of: string;
      showing: string;
      models: string;
    };
    status: {
      noMatch: string;
      noDatasets: string;
    };
  };
  /** Contact form, opened as a dialog from the home-page CTA. */
  contact: {
    cta: string;
    heading: string;
    body: string;
    fields: { name: string; email: string; message: string };
    submit: string;
    sending: string;
    successTitle: string;
    successBody: string;
    error: string;
  };
  /** Standalone /join community membership page + form. */
  join: {
    title: string;
    lede: string;
    intro: string;
    fields: {
      name: string;
      email: string;
      affiliation: string;
      role: string;
      interests: string;
      link: string;
      motivation: string;
    };
    placeholders: { affiliation: string; link: string; motivation: string };
    optional: string;
    rolePlaceholder: string;
    roleOptions: string[];
    interestOptions: string[];
    submit: string;
    sending: string;
    successTitle: string;
    successBody: string;
    error: string;
  };
  /** Cross-cutting form microcopy shared by both forms. */
  forms: { required: string; invalidEmail: string };
  post: { backResearch: string; backNews: string; readMore: string };
  footer: {
    tagline: string;
    sections: { title: string; links: { label: string; href: string }[] }[];
    rights: string;
    openSource: string;
  };
}

const en: Dictionary = {
  nav: { tseval: "TS-Eval", research: "Research", news: "News", join: "Join us" },
  themeToggle: { toLight: "Switch to light mode", toDark: "Switch to dark mode" },
  hero: {
    eyebrow: "Transparent, open-source infrastructure for AI research",
    title: "The open infrastructure AI research",
    titleAccent: "depends on.",
    lede: "Diaugeia.AI builds the foundational infrastructure that modern AI research relies on — so every result is reproducible, every tool is open, and researchers can stay at the frontier.",
    ctaNews: "Read the news",
    ctaGithub: "View on GitHub",
    etymology: "διαύγεια · Greek — clarity, lucidity, transparency",
  },
  mission: {
    label: "Mission",
    heading:
      "We build the open infrastructure that AI research depends on.",
    body: "The hardest part of an experiment is rarely the idea — it is the infrastructure beneath it. Diaugeia provides that groundwork as an open, community-run commons: modern by design, integrated with agents, reproducible end to end, and transparent at every layer.",
  },
  pillars: {
    label: "What we stand for",
    heading: "Built on four commitments.",
    items: [
      {
        icon: "sparkles",
        title: "Modern",
        body: "Built on cutting-edge design and continuously updated to stay at the frontier.",
      },
      {
        icon: "bot",
        title: "Agentic",
        body: "Tightly integrated with LLM agents to minimize the manual work humans need to do.",
      },
      {
        icon: "repeat",
        title: "Reproducibility",
        body: "Every result is traceable, re-runnable, and verifiable.",
      },
      {
        icon: "lock-open",
        title: "Open by default",
        body: "Transparent, auditable, and free to build upon.",
      },
    ],
  },
  latest: {
    researchLabel: "Latest research",
    newsLabel: "Latest news",
    viewAll: "View all",
    empty: "Nothing published yet — check back soon.",
  },
  cta: {
    heading: "Build the open infrastructure for AI research with us.",
    body: "Diaugeia.AI is open-source and community-run. Follow the news or get involved on GitHub.",
  },
  research: {
    title: "Research",
    lede: "Work on the infrastructure, methods, and open questions of AI research — reproducibility, agentic workflows, and open evaluation.",
  },
  news: {
    title: "News",
    lede: "Announcements, releases, and updates from Diaugeia.AI.",
  },
  tseval: {
    title: "TS-Eval",
    lede: "An open, reproducible leaderboard for time-series forecasting. Every entry is a community submission — one agent trajectory and one verified result — ranked transparently across tracks, datasets, and horizons.",
    updated: "Updated",
    submissions: "submissions",
    rankedBy: "ranked by",
    horizon: "pred",
    emptyTrack: "No submissions in this track yet — be the first.",
    tracks: {
      time_series: "Time Series",
      spatiotemporal: "Spatial Temporal",
      covariate: "Covariate",
      realtime: "RealTime",
      stock: "Stock-HS300",
      traffic_rt: "Traffic",
      air_quality: "Air Quality",
    },
    cols: { model: "Model", mse: "MSE", mae: "MAE", runs: "Runs" },
    categories: {
      commonStatic: "Common Static Dataset",
      commonStaticZh: "通用静态数据集",
      realtime: "Real-Time Dataset",
      realtimeZh: "实时数据集",
    },
    filters: {
      title: "Filters & Options",
      resetAll: "Reset all",
      searchPlaceholder: "Type dataset name...",
      datasetCategories: "Dataset Categories (8 Core)",
      display: "Display",
      metrics: "Metrics",
      options: "Options",
    },
    datasets: {
      etth1: "ETTh1",
      etth2: "ETTh2",
      ettm1: "ETTm1",
      ettm2: "ETTm2",
      traffic: "Traffic",
      solar: "Solar",
      electricity: "Electricity",
      weather: "Weather",
    },
    datasetDescriptions: {
      hourly1: "Hourly 1",
      hourly2: "Hourly 2",
      minute1: "Minute 1",
      minute2: "Minute 2",
      roadTraffic: "Road traffic",
      solarPower: "Solar power",
      powerUsage: "Power usage",
      weatherData: "Weather data",
    },
    displayOptions: {
      showMedals: "Show medals",
      showMedalsDesc: "🥇🥈🥉 for top 3",
      performanceBars: "Performance bars",
      performanceBarsDesc: "Visual indicators",
      highlightTop3: "Highlight top 3",
      highlightTop3Desc: "Accent background",
      compactMode: "Compact mode",
      compactModeDesc: "Reduced spacing",
      submissionIds: "Submission IDs",
      submissionIdsDesc: "Run identifiers",
      colorByRank: "Color by rank",
      colorByRankDesc: "Gradient coloring",
    },
    metricsOptions: {
      mse: "MSE",
      mseDesc: "Mean Squared Error",
      mae: "MAE",
      maeDesc: "Mean Absolute Error",
      rmse: "RMSE",
      rmseDesc: "Root Mean Squared Error",
      corr: "Corr",
      corrDesc: "Correlation coefficient",
      fitTime: "Fit Time",
      fitTimeDesc: "Training time (seconds)",
      inferenceTime: "Inference Time",
      inferenceTimeDesc: "Prediction time (seconds)",
      runs: "Runs",
      runsDesc: "Number of runs",
    },
    modelTypes: {
      title: "Models",
      timeSeries: "Time Series Models",
      timeSeriesDesc: "108 models",
      spatialTemporal: "Spatial Temporal Models",
      spatialTemporalDesc: "28 models",
    },
    viewModes: {
      table: "Table",
      compact: "Compact",
      detailed: "Detailed",
    },
    navigation: {
      previous: "Previous",
      next: "Next",
      dataset: "Dataset",
      of: "of",
      showing: "Showing",
      models: "models",
    },
    status: {
      noMatch: 'No datasets match',
      noDatasets: "No datasets in this track",
    },
  },
  contact: {
    cta: "Contact us",
    heading: "Questions, ideas, or partnerships?",
    body: "Send us a note and we'll get back to you by email.",
    fields: { name: "Name", email: "Email", message: "Message" },
    submit: "Send",
    sending: "Sending…",
    successTitle: "Message sent.",
    successBody: "Thanks — we'll reply by email soon.",
    error:
      "Something went wrong. Please try again, or email contact@diaugeia.ai directly.",
  },
  join: {
    title: "Join us",
    lede: "Help build the open infrastructure AI research depends on.",
    intro:
      "Diaugeia is an open, community-run project. Tell us a bit about yourself and how you'd like to take part — we'll follow up by email about next steps.",
    fields: {
      name: "Name",
      email: "Email",
      affiliation: "School / Affiliation",
      role: "Role",
      interests: "Areas of interest",
      link: "GitHub / Homepage",
      motivation: "Why do you want to join?",
    },
    placeholders: {
      affiliation: "University, lab, company, or “Independent”",
      link: "https://github.com/…",
      motivation:
        "What draws you to Diaugeia, and how would you like to contribute?",
    },
    optional: "(optional)",
    rolePlaceholder: "Select one",
    roleOptions: [
      "Undergraduate",
      "Master's / PhD student",
      "Researcher / Faculty",
      "Engineer / Developer",
      "Independent / Hobbyist",
      "Other",
    ],
    interestOptions: [
      "Reproducibility",
      "Agentic workflows",
      "Open evaluation & benchmarks",
      "Infrastructure & tooling",
      "Other",
    ],
    submit: "Submit application",
    sending: "Submitting…",
    successTitle: "Application received.",
    successBody:
      "Thanks for your interest — we'll be in touch by email about next steps.",
    error:
      "Something went wrong. Please try again, or email contact@diaugeia.ai directly.",
  },
  forms: {
    required: "This field is required.",
    invalidEmail: "Please enter a valid email.",
  },
  post: {
    backResearch: "All research",
    backNews: "All news",
    readMore: "Read more",
  },
  footer: {
    tagline: "Transparent, open-source infrastructure for AI research.",
    sections: [
      {
        title: "Explore",
        links: [
          { label: "Research", href: "/research" },
          { label: "News", href: "/news" },
        ],
      },
      {
        title: "Community",
        links: [
          { label: "GitHub", href: "https://github.com/Diaugeia" },
          { label: "Join us", href: "/join" },
        ],
      },
    ],
    rights: "All rights reserved.",
    openSource: "Open source · community-run",
  },
};

const zh: Dictionary = {
  nav: { tseval: "TS-Eval", research: "研究", news: "新闻", join: "加入我们" },
  themeToggle: { toLight: "切换到白天模式", toDark: "切换到暗夜模式" },
  hero: {
    eyebrow: "面向 AI 研究的透明、开源基础设施",
    title: "AI 研究所依赖的",
    titleAccent: "开放基础设施。",
    lede: "Diaugeia.AI 构建现代 AI 研究所依赖的底层基础设施——让每个结果都可复现、每件工具都开放，让研究者始终处于前沿。",
    ctaNews: "查看新闻",
    ctaGithub: "前往 GitHub",
    etymology: "διαύγεια · 希腊语 — 清澈、澄明、透明",
  },
  mission: {
    label: "使命",
    heading: "我们构建 AI 研究所依赖的开放基础设施。",
    body: "一项实验最难的部分往往不是想法，而是其下的基础设施。Diaugeia 以开放、社区运营的「公共资源」形式提供这一地基：以现代设计为本、与智能体深度集成、端到端可复现，并在每一层都保持透明。",
  },
  pillars: {
    label: "我们的主张",
    heading: "建立在四项承诺之上。",
    items: [
      {
        icon: "sparkles",
        title: "现代",
        body: "基于前沿设计构建，并持续更新以紧跟前沿。",
      },
      {
        icon: "bot",
        title: "智能体驱动",
        body: "与 LLM 智能体深度集成，尽可能减少人工操作。",
      },
      {
        icon: "repeat",
        title: "可复现",
        body: "每个结果都可追溯、可重跑、可验证。",
      },
      {
        icon: "lock-open",
        title: "默认开放",
        body: "透明、可审计，并可自由地在其之上构建。",
      },
    ],
  },
  latest: {
    researchLabel: "最新研究",
    newsLabel: "最新动态",
    viewAll: "查看全部",
    empty: "暂无内容，敬请期待。",
  },
  cta: {
    heading: "与我们一起构建 AI 研究的开放基础设施。",
    body: "Diaugeia.AI 开源且由社区运营。关注新闻，或在 GitHub 参与共建。",
  },
  research: {
    title: "研究",
    lede: "围绕 AI 研究的基础设施、方法与开放问题——可复现、智能体工作流，以及开放评测。",
  },
  news: {
    title: "新闻",
    lede: "来自 Diaugeia.AI 的公告、发布与动态。",
  },
  tseval: {
    title: "TS-Eval",
    lede: "开放、可复现的时间序列预测排行榜。每一条记录都是社区贡献——一条智能体轨迹，一个已验证的结果——透明地在赛道、数据集和预测步长之间排名。",
    updated: "更新于",
    submissions: "次提交",
    rankedBy: "排名依据",
    horizon: "pred",
    emptyTrack: "此赛道暂无提交——成为第一个。",
    tracks: {
      time_series: "时间序列",
      spatiotemporal: "时空",
      covariate: "协变量",
      realtime: "实时",
      stock: "Stock-HS300",
      traffic_rt: "交通",
      air_quality: "空气质量",
    },
    cols: { model: "模型", mse: "MSE", mae: "MAE", runs: "运行次数" },
    categories: {
      commonStatic: "Common Static Dataset",
      commonStaticZh: "通用静态数据集",
      realtime: "Real-Time Dataset",
      realtimeZh: "实时数据集",
    },
    filters: {
      title: "筛选与选项",
      resetAll: "重置全部",
      searchPlaceholder: "输入数据集名称...",
      datasetCategories: "数据集类别（8个核心）",
      display: "显示",
      metrics: "度量指标",
      options: "选项",
    },
    datasets: {
      etth1: "ETTh1",
      etth2: "ETTh2",
      ettm1: "ETTm1",
      ettm2: "ETTm2",
      traffic: "Traffic",
      solar: "Solar",
      electricity: "Electricity",
      weather: "Weather",
    },
    datasetDescriptions: {
      hourly1: "小时级 1",
      hourly2: "小时级 2",
      minute1: "分钟级 1",
      minute2: "分钟级 2",
      roadTraffic: "道路交通",
      solarPower: "太阳能",
      powerUsage: "电力使用",
      weatherData: "天气数据",
    },
    displayOptions: {
      showMedals: "显示奖牌",
      showMedalsDesc: "前 3 名显示 🥇🥈🥉",
      performanceBars: "性能条形图",
      performanceBarsDesc: "可视化指标",
      highlightTop3: "高亮前 3 名",
      highlightTop3Desc: "强调背景色",
      compactMode: "紧凑模式",
      compactModeDesc: "减少间距",
      submissionIds: "提交 ID",
      submissionIdsDesc: "运行标识符",
      colorByRank: "按排名着色",
      colorByRankDesc: "渐变色彩",
    },
    metricsOptions: {
      mse: "MSE",
      mseDesc: "均方误差",
      mae: "MAE",
      maeDesc: "平均绝对误差",
      rmse: "RMSE",
      rmseDesc: "均方根误差",
      corr: "Corr",
      corrDesc: "相关系数",
      fitTime: "训练时间",
      fitTimeDesc: "训练时间（秒）",
      inferenceTime: "推理时间",
      inferenceTimeDesc: "预测时间（秒）",
      runs: "运行次数",
      runsDesc: "实验运行次数",
    },
    modelTypes: {
      title: "模型",
      timeSeries: "时间序列模型",
      timeSeriesDesc: "108 个模型",
      spatialTemporal: "时空模型",
      spatialTemporalDesc: "28 个模型",
    },
    viewModes: {
      table: "表格",
      compact: "紧凑",
      detailed: "详细",
    },
    navigation: {
      previous: "上一个",
      next: "下一个",
      dataset: "数据集",
      of: "/",
      showing: "显示",
      models: "个模型",
    },
    status: {
      noMatch: '没有匹配的数据集',
      noDatasets: "此赛道暂无数据集",
    },
  },
  contact: {
    cta: "联系我们",
    heading: "有问题、想法或合作意向？",
    body: "给我们留言，我们会通过邮件回复你。",
    fields: { name: "姓名", email: "邮箱", message: "留言" },
    submit: "发送",
    sending: "发送中…",
    successTitle: "已发送。",
    successBody: "谢谢——我们会尽快通过邮件回复你。",
    error: "出了点问题，请重试，或直接发邮件至 contact@diaugeia.ai。",
  },
  join: {
    title: "加入我们",
    lede: "一起构建 AI 研究所依赖的开放基础设施。",
    intro:
      "Diaugeia 是一个开放、由社区运营的项目。简单介绍一下你自己，以及你希望以怎样的方式参与——我们会通过邮件就后续步骤与你联系。",
    fields: {
      name: "姓名",
      email: "邮箱",
      affiliation: "学校 / 机构",
      role: "身份",
      interests: "感兴趣的方向",
      link: "GitHub / 个人主页",
      motivation: "你为何想加入？",
    },
    placeholders: {
      affiliation: "大学、实验室、公司，或填「独立」",
      link: "https://github.com/…",
      motivation: "是什么吸引你加入 Diaugeia，你希望如何参与贡献？",
    },
    optional: "（选填）",
    rolePlaceholder: "请选择",
    roleOptions: [
      "本科生",
      "硕士 / 博士生",
      "研究员 / 教师",
      "工程师 / 开发者",
      "独立研究者 / 爱好者",
      "其他",
    ],
    interestOptions: [
      "可复现性",
      "智能体工作流",
      "开放评测与基准",
      "基础设施与工具",
      "其他",
    ],
    submit: "提交申请",
    sending: "提交中…",
    successTitle: "已收到你的申请。",
    successBody: "感谢你的关注——我们会通过邮件就后续步骤与你联系。",
    error: "出了点问题，请重试，或直接发邮件至 contact@diaugeia.ai。",
  },
  forms: {
    required: "此项为必填。",
    invalidEmail: "请输入有效的邮箱地址。",
  },
  post: {
    backResearch: "全部研究",
    backNews: "全部新闻",
    readMore: "阅读全文",
  },
  footer: {
    tagline: "面向 AI 研究的透明、开源基础设施。",
    sections: [
      {
        title: "浏览",
        links: [
          { label: "研究", href: "/research" },
          { label: "新闻", href: "/news" },
        ],
      },
      {
        title: "社区",
        links: [
          { label: "GitHub", href: "https://github.com/Diaugeia" },
          { label: "加入我们", href: "/join" },
        ],
      },
    ],
    rights: "保留所有权利。",
    openSource: "开源 · 社区运营",
  },
};

const dictionaries: Record<Locale, Dictionary> = { en, zh };

export function getDictionary(locale: Locale): Dictionary {
  return dictionaries[locale];
}
