## What is a Convolution?

A convolution is a mathematical operation used in CNNs to extract features from input data, such as images. In the context of image processing, a convolution involves sliding a small matrix, called a kernel or filter, over the input image and computing the dot product between the kernel and the overlapping region of the input image.

## What is Max Pooling? Average Pooling?

Max Pooling and Average Pooling are downsampling techniques used to reduce the spatial dimensions (width and height) of the input volume.

- Max Pooling: Selects the maximum value from the region covered by the filter. This helps to capture the most prominent features.

- Average Pooling: Computes the average value of the elements in the region covered by the filter. This helps to smooth the features.

## What is a Kernel/Filter?

A kernel or filter is a small matrix used in the convolution operation. It slides over the input data to produce a feature map. Kernels are used to detect different features such as edges, textures, and patterns.

## What is Padding?

Padding is the process of adding extra pixels to the input image's borders. It helps control the spatial size of the output feature maps. Padding can be done in two ways:

## What is “Same” Padding? “Valid” Padding?

- Same Padding: Adds zeros to the input image's borders so that the output feature map has the same spatial dimensions as the input. It ensures that the kernel can be applied to all elements of the input image.

- Valid Padding: No padding is applied, and the kernel is only applied to valid positions inside the input image. This results in a smaller output feature map.

## What is a Stride?

The stride is the number of pixels by which the kernel moves across the input image. A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means the kernel moves two pixels at a time.

## What are Channels?

Channels refer to the number of color components in an image. For example, a color image has three channels (Red, Green, Blue), while a grayscale image has one channel.

## How to Perform a Convolution Over an Image

Performing a convolution over an image involves the following steps:

1. Place the kernel at the top-left corner of the image.
2. Compute the dot product between the kernel and the overlapping region of the image.
3. Move the kernel to the next position based on the stride and repeat the dot product calculation.
4. Continue this process until the kernel has slid over the entire image.

Example in Python:

    import numpy as np

    def convolve(image, kernel, stride=1, padding='valid'):
        if padding == 'same':
            pad_h = (kernel.shape[0] - 1) // 2
            pad_w = (kernel.shape[1] - 1) // 2
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        elif padding == 'valid':
            pad_h = pad_w = 0

        output_shape = ((image.shape[0] - kernel.shape[0] + 2 * pad_h) // stride + 1,
                        (image.shape[1] - kernel.shape[1] + 2 * pad_w) // stride + 1)

        output = np.zeros(output_shape)
        for y in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
            for x in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
                output[y // stride, x // stride] = np.sum(image[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)

        return output

    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 0], [0, -1]])
    print(convolve(image, kernel, stride=1, padding='valid'))

## How to Perform Max/Average Pooling Over an Image

Max Pooling Example in Python:

    def max_pooling(image, size=2, stride=2):
        output_shape = ((image.shape[0] - size) // stride + 1,
                        (image.shape[1] - size) // stride + 1)

        output = np.zeros(output_shape)
        for y in range(0, image.shape[0] - size + 1, stride):
            for x in range(0, image.shape[1] - size + 1, stride):
                output[y // stride, x // stride] = np.max(image[y:y + size, x:x + size])

        return output

    print(max_pooling(image, size=2, stride=2))

Average Pooling Example in Python:

    def average_pooling(image, size=2, stride=2):
        output_shape = ((image.shape[0] - size) // stride + 1,
                        (image.shape[1] - size) // stride + 1)

        output = np.zeros(output_shape)
        for y in range(0, image.shape[0] - size + 1, stride):
            for x in range(0, image.shape[1] - size + 1, stride):
                output[y // stride, x // stride] = np.mean(image[y:y + size, x:x + size])

        return output

    print(average_pooling(image, size=2, stride=2))

These examples illustrate how to perform convolution and pooling operations over images, which are essential techniques in building convolutional neural networks for image processing tasks.
