# Wasserstein Style Distribution Analysis and Transform  for Stylized Image Generation
This is the code of paper Wasserstein Style Distribution Analysis and Transform  for Stylized Image Generation. Xi Yu, Xiang Gu, Zhihao Shi, Jian Sun. ICCV 2025.

![Image text](visualization.png)

##  🧾 1.  Abstract

Large-scale text-to-image diffusion models have achieved remarkable success in image generation, thereby driving the development of stylized image generation technologies. Recent studies introduce style information by empirically replacing specific features in attention blocks with style features. However, the relationship between features and style remains unclear. In this paper, we systematically analyze the relationship between features in attention blocks and style. By quantifying the distribution discrepancy induced by style variations using the Wasserstein distance, we find that features in self-attention blocks exhibit high sensitivity to style compared to features in cross-attention blocks. Our analysis provides valuable insights into the contribution of different features to style. Based on our findings, we propose a novel Wasserstein Style Distribution Transform (WSDT) method, which generates stylized images by transforming the distribution of style-sensitive features to align with that of style features. WSDT applies channel adaptive distribution transform to ensure that information not related to the style is not introduced. Our approach is simple yet efficient, optimization-free, and can be seamlessly integrated into attention-based text-to-image diffusion models. Extensive experiments demonstrate the effectiveness of our approach in stylized image generation tasks.

