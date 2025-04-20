# Temporal Fusion Transformer (TFT): Implementation and Optimization Using Optuna
Medium Post: [https://medium.com/@angelAjcabul/temporal-fusion-transformer-tft-implementation-and-optimization-using-optuna-d87c8aacfb3a]
## Multi-Horizon Forecasting

Multi-horizon time series forecasting is one of the most critical challenges in machine learning, as it enables the projection of future scenarios based on historical patterns and exogenous variables. Its applications span a wide range of industries—from retail sales forecasting and food production planning to more routine aspects such as price fluctuations of fruits and vegetables in the market.
![TFT Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*j5GK3ITDvsDEQm7kVzoO_w.png)


In practice, forecasting across multiple temporal horizons can be complex, particularly when incorporating historical variables (e.g., weather conditions, website traffic, raw material prices), time-invariant metadata related to the system under analysis (e.g., location, product category, store type), and known future variables (e.g., scheduled special events, concerts, temporary closures on specific dates).

## The Evolution of Other Models

Deep Neural Networks (DNNs) have shown significant improvements in multi-horizon time series forecasting, often outperforming traditional models. More recently, the integration of attention-based architectures, such as Transformer models, has further enhanced the ability to capture key temporal patterns, thereby improving forecast accuracy.

However, these models still present certain limitations that must be addressed:

- **Data heterogeneity**: The variability in input features makes consistent modeling across different time series challenging.  
- **Assumption of future exogenous variables**: Many models assume future exogenous inputs are fully known, which is often unrealistic in autoregressive settings.  
- **Handling of static variables**: Instead of treating static covariates separately, they are commonly concatenated with temporal features at each timestep, potentially reducing model efficiency.  
- **Lack of interpretability**: The complex, non-linear interactions among multiple parameters often render these models as black boxes, limiting their usability and diagnosability.

To address these challenges, a novel attention-based architecture was proposed: the **Temporal Fusion Transformer (TFT)**. This model not only improves forecast performance but also enhances interpretability. In the following section, we analyze its architecture, starting with the core forecasting equation for time series modeling.

---

## TFT Architecture Overview

With a clearer understanding of the base forecasting equation, it is possible to delve into the architecture that enables the Temporal Fusion Transformer (TFT) to operate effectively. The following provides an overview of its key components, though a more in-depth explanation is available in the [original paper](https://arxiv.org/abs/1912.09363).
![TFT Model Formula](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*77WyrUknM9W-rEOVh_ocRg.png)



### Block 1 — Known Future Inputs

This part processes both historical data and known future inputs using an LSTM-based decoder, capable of capturing long-term dependencies and generating coherent output sequences.

The **Add & Norm** layer stabilizes gradient flow by normalizing activations, while skip connections preserve critical original features, enhancing predictive performance.
![TFT Model Formula](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*77WyrUknM9W-rEOVh_ocRg.png)

### Block 2 — Past Inputs

This block mirrors the structure of Block 1 but uses an **LSTM Encoder** to capture dependencies from historical data. The encoder compresses these patterns into a context vector used for downstream predictions. Unlike the decoder, the encoder focuses solely on encoding past information, which, in combination with future covariates, improves forecast accuracy.
![TFT Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*beXkJmmvs_ryXr86eLw22w.png)

### Block 3 — Gated Residual Network (GRN)

The GRN selectively filters relevant information using gating mechanisms. This prevents noisy or redundant inputs from flowing freely through the model, promoting more efficient learning and preserving important signals.
![TFT Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GG5WRiSY5G6ym6Leqql6Tw.png)

### Block 4 — Masked Interpretable Multi-Head Attention (MIMA)

This module improves the model's ability to capture complex dependencies by focusing attention on relevant segments of the input. Each attention head captures different aspects (local, global, or contextual), enhancing the richness of the learned representation. In the architecture, arrows indicate how the outputs from GRNs are fed into multiple attention heads in parallel.
![TFT Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Tl4fCHH53UXVGA-RiRCl-w.png)

### Block 5 — Add & Norm  GRN

After MIMA, outputs are passed through additional Add & Norm layers to stabilize training and integrate residual information. Skip connections from GRNs preserve essential signals, while additional GRNs filter noise, ensuring only the most informative content continues forward.

---
![TFT Model](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*lnuZiI2qg6ZqCv-q2Y8pHw.png)
## Temporal Fusion Transformer (TFT) Architecture Summary

Up to this point, we have described the architecture of the Temporal Fusion Transformer (TFT), highlighting how it integrates both historical and known future variables across multiple temporal resolutions. For a deeper analysis of its internal structure and components, it is recommended to consult the original paper.

---
