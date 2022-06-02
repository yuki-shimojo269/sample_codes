# sample_codes
お試しコード

# Pytorch
pytorchに関する困ったことや、やりたいことのサンプルコードを乗っける

## 01_save_model
モデルの重みを保存する。このときEncoderとdecoderで保存するファイルを分ける
```train.py
torch.save(self.net.Encoder.state_dict(), 'Net_encoder.pth')
torch.save(self.net.Decoder.state_dict(), 'Net_decoder.pth')
```
```test.py
self.net.Encoder.load_state_dict(torch.load('Net_encoder.pth'))
self.net.Decoder.load_state_dict(torch.load('Net_decoder.pth'))
```