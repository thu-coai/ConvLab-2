# DAMD

DAMD is an end-to-end model designed to handle the multi-domain response generation problem through leveraging the proposed multi-action data augmentation framework. There are one encoder that encodes dialog context and three decoders that decodes belief span, system action span and system response respectively.

We implement the inference for deployment here. Please refer to [github](https://github.com/thu-spmi/damd-multiwoz) for training and other aspects. The original paper can be found at [arxiv](https://arxiv.org/abs/1911.10484).

## Reference

   ```
@inproceedings{zhang2020task,
  title={Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses under the Same Context},
  author={Zhang, Yichi and Ou, Zhijian and Yu, Zhou},
  booktitle={34th AAAI Conference on Artificial Intelligence},
  year={2020}
}
   ```

