from evalue import predict_image

while True:
    test_image = input("图片路径:")
    print(predict_image(test_image,threshold=0.85))