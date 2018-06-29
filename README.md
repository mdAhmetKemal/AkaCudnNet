  # AkaCudnNet
  Basic Neural Network Framework  using CUDNN and CUBLAS
  CUDNN ve CUBLAS kullanan basit bir framework
Yeni algoritmalarımı denemek için böyle bir frameworku geliştirmeye karar verdim.
Uygulama prototxt.ini dosyasından ağ yapısını öğrenerek ağ yapısını oluşturuyor.
Ağ yapısından sonra prototxt.ini içerisnde adı geçen train datası ile her epoch öğrenme yapıyor.
epoch sonucu prototxt.in dosyasında adı geçen test databasine göre test işlemini uygulayıp 
ekrana yansıtıyor .Amacım  bu uygulamayı daha sade kullanımı basit ve elle tutulur bir hale 
getirip insanlarla paylaşmak.

   Asıl amacım malasef burda belirtemeyeceğim kendi ağ yapılarımı algoritmalarımı geliştirmek.
Bu sebeple kaynak kodları veremeyeceğim.
  
  Uygulama test ve train dataseti olarak Mnist,cifar10,cifar100  dataları ile çalışabilecek şekilde 
geliştirildi.
  
Içerik olarak Convolution2d ,poolMax,poolAvg,batchNormalization,FullyConnect,Dropuot,Add,
Concantrate,Activation(relu,tanh,sigm,crelu,elu),Direct  bağlantılarına sahip.
Adı geçen bağlantı tiplerinden bağzılarını prototxt.ini dosyası içerinde açıklamaları göreceksiniz.

  Yazılım sadece Nvidia Ekran Kartlarında çalışıyor (compute capability 3.5 ve üzeri)
Yükleme için linkte verdiğim DLL dosyalarını yazılımın olduğu klasöre kopyalamanız yeterli
(Cuda ,cublas ,cudnnni ayrıca sisteme yüklemenize gerek yok )

Ayrıca prototxt.ini dosyası içerisinde geçen datasetleri de linkte verdim.
Aynı klasöre koyarak test yapabilirsiniz.

Dll dosyaları https://yadi.sk/d/EZhfYrwt3Ybks4

Dataset dosyaları https://yadi.sk/d/RvTJRN_y3Ybnfh
çalıştırmak için ilgili dosyaları programın olduğu klasöre koyduktan sonra,runScript ile çalıştırabilirsiniz.

![alt text](https://github.com/mdAhmetKemal/AkaCudnNet/blob/master/setup.jpg) 
