# Contributing
Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Getting Started

Setting up `TextSentimentAnalysis` for local development.

1. Fork the [TextSentimentAnalysis](https://github.com/khanfarhan10/TextSentimentAnalysis) repo on GitHub.

2. Clone your fork locally:

```bash
$ git clone https://github.com/your_username_here/TextSentimentAnalysis
```

3. Add a tracking branch which can always have the latest version of TextSentimentAnalysis.

```bash
$ git remote add TextSentimentAnalysis https://github.com/khanfarhan10/TextSentimentAnalysis
$ git fetch TextSentimentAnalysis
$ git branch TextSentimentAnalysis-master --track TextSentimentAnalysis/master
$ git checkout TextSentimentAnalysis
$ git pull
```

4. Create a branch from the last dev version of your tracking branch for local development:

```bash
$ git checkout -b name-of-your-bugfix-or-feature
```

5. Install it locally:
   * Check Setting up section in [README.md](README.md)

6. Now make your changes locally:

```bash
$ git add .
$ git commit -m "Your detailed description of your changes."
$ git push -u origin name-of-your-bugfix-or-feature
```

7. Submit a Pull Request on Github to the `master` branch.
