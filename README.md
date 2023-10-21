page-flip-detector
==============================

# Data Description:

We collected page-flipping videos from smartphones and labeled them as flipping and not flipping.

We clipped the videos as short videos and labeled them as flipping or not flipping. The extracted frames are then saved to disk in sequential order with the following naming structure: VideoID_FrameNumber

# Goal(s):

Predict if the page is being flipped using a single image.

Success Metrics:


Evaluate model performance based on F1 score, the higher the better.

# Project Development

# Image Classification with PyTorch

This Jupyter Notebook contains the code for training and evaluating image classification models using PyTorch. 

## Development

The notebook starts by importing the necessary libraries and loading the dataset. The dataset consists of images of pages being flipped or not, which are split into training and validation sets. The notebook then defines and trains two different models: cnn_model and mobilenet_v2. The first is built from scratch using the Pytorch nn module. The Mobile Net is a well-known mobile and light model, and we apply transfer learning to it.  After training both models on our dataset, we found that cnn_model performed better than mobilenet_v2, achieving an F1 score of 97.5%. This indicates that cnn_model is a good candidate for further testing and deployment.

## Conclusion

In this phase of testing, we trained and evaluated two different models: cnn_model and mobilenet_v2. After training both models on our dataset, we found that cnn_model performed better than mobilenet_v2, achieving an F1 score of 97.5%. This indicates that cnn_model is a good candidate for further testing and deployment. However, it's important to note that this F1 score was achieved on a specific dataset and may not generalize well to other datasets. Therefore, further testing and evaluation are necessary before deploying this model in a production environment.

## Results

The best model was the 'cnn_model' with Train Loss: 0.037 Train Accuracy: 0.99 Validation Loss: 0.021 Validation Accuracy: 0.99. We also made predictions on the test set and computed the F1 score, which was 97.5%.

![output](https://github.com/joaothomazlemos/page-flip-detector/assets/62029505/3159cadb-0185-4b0d-9443-5a0601199e6d)

![Uploading<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="943.78125pt" height="488.27625pt" viewBox="0 0 943.78125 488.27625" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2023-10-20T19:50:06.344176</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.8.0, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 488.27625 
L 943.78125 488.27625 
L 943.78125 0 
L 0 0 
z
" style="fill: #ffffff"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 43.78125 450.72 
L 936.58125 450.72 
L 936.58125 7.2 
L 43.78125 7.2 
z
" style="fill: #ffffff"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <defs>
       <path id="m06e316a231" d="M 0 0 
L 0 3.5 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m06e316a231" x="84.363068" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- 0.0 -->
      <g transform="translate(76.411506 465.318437) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-30" d="M 2034 4250 
Q 1547 4250 1301 3770 
Q 1056 3291 1056 2328 
Q 1056 1369 1301 889 
Q 1547 409 2034 409 
Q 2525 409 2770 889 
Q 3016 1369 3016 2328 
Q 3016 3291 2770 3770 
Q 2525 4250 2034 4250 
z
M 2034 4750 
Q 2819 4750 3233 4129 
Q 3647 3509 3647 2328 
Q 3647 1150 3233 529 
Q 2819 -91 2034 -91 
Q 1250 -91 836 529 
Q 422 1150 422 2328 
Q 422 3509 836 4129 
Q 1250 4750 2034 4750 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-2e" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_2">
      <g>
       <use xlink:href="#m06e316a231" x="191.157327" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- 2.5 -->
      <g transform="translate(183.205764 465.318437) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-32" d="M 1228 531 
L 3431 531 
L 3431 0 
L 469 0 
L 469 531 
Q 828 903 1448 1529 
Q 2069 2156 2228 2338 
Q 2531 2678 2651 2914 
Q 2772 3150 2772 3378 
Q 2772 3750 2511 3984 
Q 2250 4219 1831 4219 
Q 1534 4219 1204 4116 
Q 875 4013 500 3803 
L 500 4441 
Q 881 4594 1212 4672 
Q 1544 4750 1819 4750 
Q 2544 4750 2975 4387 
Q 3406 4025 3406 3419 
Q 3406 3131 3298 2873 
Q 3191 2616 2906 2266 
Q 2828 2175 2409 1742 
Q 1991 1309 1228 531 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-35" d="M 691 4666 
L 3169 4666 
L 3169 4134 
L 1269 4134 
L 1269 2991 
Q 1406 3038 1543 3061 
Q 1681 3084 1819 3084 
Q 2600 3084 3056 2656 
Q 3513 2228 3513 1497 
Q 3513 744 3044 326 
Q 2575 -91 1722 -91 
Q 1428 -91 1123 -41 
Q 819 9 494 109 
L 494 744 
Q 775 591 1075 516 
Q 1375 441 1709 441 
Q 2250 441 2565 725 
Q 2881 1009 2881 1497 
Q 2881 1984 2565 2268 
Q 2250 2553 1709 2553 
Q 1456 2553 1204 2497 
Q 953 2441 691 2322 
L 691 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-32"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-35" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_3">
      <g>
       <use xlink:href="#m06e316a231" x="297.951585" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- 5.0 -->
      <g transform="translate(290.000022 465.318437) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-35"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_4">
      <g>
       <use xlink:href="#m06e316a231" x="404.745843" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- 7.5 -->
      <g transform="translate(396.794281 465.318437) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-37" d="M 525 4666 
L 3525 4666 
L 3525 4397 
L 1831 0 
L 1172 0 
L 2766 4134 
L 525 4134 
L 525 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-37"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-35" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_5">
      <g>
       <use xlink:href="#m06e316a231" x="511.540102" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- 10.0 -->
      <g transform="translate(500.407289 465.318437) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-31" d="M 794 531 
L 1825 531 
L 1825 4091 
L 703 3866 
L 703 4441 
L 1819 4666 
L 2450 4666 
L 2450 531 
L 3481 531 
L 3481 0 
L 794 0 
L 794 531 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-30" x="63.623047"/>
       <use xlink:href="#DejaVuSans-2e" x="127.246094"/>
       <use xlink:href="#DejaVuSans-30" x="159.033203"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_6">
      <g>
       <use xlink:href="#m06e316a231" x="618.33436" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- 12.5 -->
      <g transform="translate(607.201548 465.318437) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-32" x="63.623047"/>
       <use xlink:href="#DejaVuSans-2e" x="127.246094"/>
       <use xlink:href="#DejaVuSans-35" x="159.033203"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_7">
      <g>
       <use xlink:href="#m06e316a231" x="725.128618" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- 15.0 -->
      <g transform="translate(713.995806 465.318437) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-35" x="63.623047"/>
       <use xlink:href="#DejaVuSans-2e" x="127.246094"/>
       <use xlink:href="#DejaVuSans-30" x="159.033203"/>
      </g>
     </g>
    </g>
    <g id="xtick_8">
     <g id="line2d_8">
      <g>
       <use xlink:href="#m06e316a231" x="831.922877" y="450.72" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- 17.5 -->
      <g transform="translate(820.790064 465.318437) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-37" x="63.623047"/>
       <use xlink:href="#DejaVuSans-2e" x="127.246094"/>
       <use xlink:href="#DejaVuSans-35" x="159.033203"/>
      </g>
     </g>
    </g>
    <g id="text_9">
     <!-- Epoch -->
     <g transform="translate(474.870312 478.996562) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-45" d="M 628 4666 
L 3578 4666 
L 3578 4134 
L 1259 4134 
L 1259 2753 
L 3481 2753 
L 3481 2222 
L 1259 2222 
L 1259 531 
L 3634 531 
L 3634 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-70" d="M 1159 525 
L 1159 -1331 
L 581 -1331 
L 581 3500 
L 1159 3500 
L 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
z
M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6f" d="M 1959 3097 
Q 1497 3097 1228 2736 
Q 959 2375 959 1747 
Q 959 1119 1226 758 
Q 1494 397 1959 397 
Q 2419 397 2687 759 
Q 2956 1122 2956 1747 
Q 2956 2369 2687 2733 
Q 2419 3097 1959 3097 
z
M 1959 3584 
Q 2709 3584 3137 3096 
Q 3566 2609 3566 1747 
Q 3566 888 3137 398 
Q 2709 -91 1959 -91 
Q 1206 -91 779 398 
Q 353 888 353 1747 
Q 353 2609 779 3096 
Q 1206 3584 1959 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-63" d="M 3122 3366 
L 3122 2828 
Q 2878 2963 2633 3030 
Q 2388 3097 2138 3097 
Q 1578 3097 1268 2742 
Q 959 2388 959 1747 
Q 959 1106 1268 751 
Q 1578 397 2138 397 
Q 2388 397 2633 464 
Q 2878 531 3122 666 
L 3122 134 
Q 2881 22 2623 -34 
Q 2366 -91 2075 -91 
Q 1284 -91 818 406 
Q 353 903 353 1747 
Q 353 2603 823 3093 
Q 1294 3584 2113 3584 
Q 2378 3584 2631 3529 
Q 2884 3475 3122 3366 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-68" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 4863 
L 1159 4863 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-45"/>
      <use xlink:href="#DejaVuSans-70" x="63.183594"/>
      <use xlink:href="#DejaVuSans-6f" x="126.660156"/>
      <use xlink:href="#DejaVuSans-63" x="187.841797"/>
      <use xlink:href="#DejaVuSans-68" x="242.822266"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_9">
      <defs>
       <path id="m633095f8ad" d="M 0 0 
L -3.5 0 
" style="stroke: #000000; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="435.869768" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- 0.0 -->
      <g transform="translate(20.878125 439.668986) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_10">
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="353.789086" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- 0.2 -->
      <g transform="translate(20.878125 357.588305) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-32" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_11">
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="271.708405" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_12">
      <!-- 0.4 -->
      <g transform="translate(20.878125 275.507623) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-34" d="M 2419 4116 
L 825 1625 
L 2419 1625 
L 2419 4116 
z
M 2253 4666 
L 3047 4666 
L 3047 1625 
L 3713 1625 
L 3713 1100 
L 3047 1100 
L 3047 0 
L 2419 0 
L 2419 1100 
L 313 1100 
L 313 1709 
L 2253 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-34" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_12">
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="189.627723" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_13">
      <!-- 0.6 -->
      <g transform="translate(20.878125 193.426942) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-36" d="M 2113 2584 
Q 1688 2584 1439 2293 
Q 1191 2003 1191 1497 
Q 1191 994 1439 701 
Q 1688 409 2113 409 
Q 2538 409 2786 701 
Q 3034 994 3034 1497 
Q 3034 2003 2786 2293 
Q 2538 2584 2113 2584 
z
M 3366 4563 
L 3366 3988 
Q 3128 4100 2886 4159 
Q 2644 4219 2406 4219 
Q 1781 4219 1451 3797 
Q 1122 3375 1075 2522 
Q 1259 2794 1537 2939 
Q 1816 3084 2150 3084 
Q 2853 3084 3261 2657 
Q 3669 2231 3669 1497 
Q 3669 778 3244 343 
Q 2819 -91 2113 -91 
Q 1303 -91 875 529 
Q 447 1150 447 2328 
Q 447 3434 972 4092 
Q 1497 4750 2381 4750 
Q 2619 4750 2861 4703 
Q 3103 4656 3366 4563 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-36" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_13">
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="107.547042" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_14">
      <!-- 0.8 -->
      <g transform="translate(20.878125 111.34626) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-38" d="M 2034 2216 
Q 1584 2216 1326 1975 
Q 1069 1734 1069 1313 
Q 1069 891 1326 650 
Q 1584 409 2034 409 
Q 2484 409 2743 651 
Q 3003 894 3003 1313 
Q 3003 1734 2745 1975 
Q 2488 2216 2034 2216 
z
M 1403 2484 
Q 997 2584 770 2862 
Q 544 3141 544 3541 
Q 544 4100 942 4425 
Q 1341 4750 2034 4750 
Q 2731 4750 3128 4425 
Q 3525 4100 3525 3541 
Q 3525 3141 3298 2862 
Q 3072 2584 2669 2484 
Q 3125 2378 3379 2068 
Q 3634 1759 3634 1313 
Q 3634 634 3220 271 
Q 2806 -91 2034 -91 
Q 1263 -91 848 271 
Q 434 634 434 1313 
Q 434 1759 690 2068 
Q 947 2378 1403 2484 
z
M 1172 3481 
Q 1172 3119 1398 2916 
Q 1625 2713 2034 2713 
Q 2441 2713 2670 2916 
Q 2900 3119 2900 3481 
Q 2900 3844 2670 4047 
Q 2441 4250 2034 4250 
Q 1625 4250 1398 4047 
Q 1172 3844 1172 3481 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-38" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_14">
      <g>
       <use xlink:href="#m633095f8ad" x="43.78125" y="25.46636" style="stroke: #000000; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_15">
      <!-- 1.0 -->
      <g transform="translate(20.878125 29.265579) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="text_16">
     <!-- Loss and Accuracy -->
     <g transform="translate(14.798438 275.340469) rotate(-90) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-4c" d="M 628 4666 
L 1259 4666 
L 1259 531 
L 3531 531 
L 3531 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-73" d="M 2834 3397 
L 2834 2853 
Q 2591 2978 2328 3040 
Q 2066 3103 1784 3103 
Q 1356 3103 1142 2972 
Q 928 2841 928 2578 
Q 928 2378 1081 2264 
Q 1234 2150 1697 2047 
L 1894 2003 
Q 2506 1872 2764 1633 
Q 3022 1394 3022 966 
Q 3022 478 2636 193 
Q 2250 -91 1575 -91 
Q 1294 -91 989 -36 
Q 684 19 347 128 
L 347 722 
Q 666 556 975 473 
Q 1284 391 1588 391 
Q 1994 391 2212 530 
Q 2431 669 2431 922 
Q 2431 1156 2273 1281 
Q 2116 1406 1581 1522 
L 1381 1569 
Q 847 1681 609 1914 
Q 372 2147 372 2553 
Q 372 3047 722 3315 
Q 1072 3584 1716 3584 
Q 2034 3584 2315 3537 
Q 2597 3491 2834 3397 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-20" transform="scale(0.015625)"/>
       <path id="DejaVuSans-61" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6e" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-64" d="M 2906 2969 
L 2906 4863 
L 3481 4863 
L 3481 0 
L 2906 0 
L 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
z
M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-41" d="M 2188 4044 
L 1331 1722 
L 3047 1722 
L 2188 4044 
z
M 1831 4666 
L 2547 4666 
L 4325 0 
L 3669 0 
L 3244 1197 
L 1141 1197 
L 716 0 
L 50 0 
L 1831 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-75" d="M 544 1381 
L 544 3500 
L 1119 3500 
L 1119 1403 
Q 1119 906 1312 657 
Q 1506 409 1894 409 
Q 2359 409 2629 706 
Q 2900 1003 2900 1516 
L 2900 3500 
L 3475 3500 
L 3475 0 
L 2900 0 
L 2900 538 
Q 2691 219 2414 64 
Q 2138 -91 1772 -91 
Q 1169 -91 856 284 
Q 544 659 544 1381 
z
M 1991 3584 
L 1991 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-72" d="M 2631 2963 
Q 2534 3019 2420 3045 
Q 2306 3072 2169 3072 
Q 1681 3072 1420 2755 
Q 1159 2438 1159 1844 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1341 3275 1631 3429 
Q 1922 3584 2338 3584 
Q 2397 3584 2469 3576 
Q 2541 3569 2628 3553 
L 2631 2963 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-79" d="M 2059 -325 
Q 1816 -950 1584 -1140 
Q 1353 -1331 966 -1331 
L 506 -1331 
L 506 -850 
L 844 -850 
Q 1081 -850 1212 -737 
Q 1344 -625 1503 -206 
L 1606 56 
L 191 3500 
L 800 3500 
L 1894 763 
L 2988 3500 
L 3597 3500 
L 2059 -325 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-4c"/>
      <use xlink:href="#DejaVuSans-6f" x="53.962891"/>
      <use xlink:href="#DejaVuSans-73" x="115.144531"/>
      <use xlink:href="#DejaVuSans-73" x="167.244141"/>
      <use xlink:href="#DejaVuSans-20" x="219.34375"/>
      <use xlink:href="#DejaVuSans-61" x="251.130859"/>
      <use xlink:href="#DejaVuSans-6e" x="312.410156"/>
      <use xlink:href="#DejaVuSans-64" x="375.789062"/>
      <use xlink:href="#DejaVuSans-20" x="439.265625"/>
      <use xlink:href="#DejaVuSans-41" x="471.052734"/>
      <use xlink:href="#DejaVuSans-63" x="537.710938"/>
      <use xlink:href="#DejaVuSans-63" x="592.691406"/>
      <use xlink:href="#DejaVuSans-75" x="647.671875"/>
      <use xlink:href="#DejaVuSans-72" x="711.050781"/>
      <use xlink:href="#DejaVuSans-61" x="752.164062"/>
      <use xlink:href="#DejaVuSans-63" x="813.443359"/>
      <use xlink:href="#DejaVuSans-79" x="868.423828"/>
     </g>
    </g>
   </g>
   <g id="line2d_15">
    <path d="M 84.363068 155.763034 
L 127.080772 170.323113 
L 169.798475 252.712947 
L 212.516178 335.740138 
L 255.233882 362.008288 
L 297.951585 383.780835 
L 340.669288 387.109366 
L 383.386992 399.136008 
L 426.104695 410.317535 
L 468.822398 412.491644 
L 511.540102 411.521541 
L 554.257805 419.252177 
L 596.975508 425.272494 
L 639.693212 424.016772 
L 682.410915 422.931808 
L 725.128618 422.383104 
L 767.846322 422.946754 
L 810.564025 428.096761 
L 853.281728 430.56 
L 895.999432 420.780492 
" clip-path="url(#pa4ece0553c)" style="fill: none; stroke: #ff0000; stroke-width: 1.5; stroke-linecap: square"/>
    <defs>
     <path id="m256b22af28" d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
" style="stroke: #ff0000"/>
    </defs>
    <g clip-path="url(#pa4ece0553c)">
     <use xlink:href="#m256b22af28" x="84.363068" y="155.763034" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="127.080772" y="170.323113" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="169.798475" y="252.712947" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="212.516178" y="335.740138" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="255.233882" y="362.008288" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="297.951585" y="383.780835" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="340.669288" y="387.109366" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="383.386992" y="399.136008" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="426.104695" y="410.317535" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="468.822398" y="412.491644" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="511.540102" y="411.521541" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="554.257805" y="419.252177" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="596.975508" y="425.272494" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="639.693212" y="424.016772" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="682.410915" y="422.931808" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="725.128618" y="422.383104" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="767.846322" y="422.946754" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="810.564025" y="428.096761" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="853.281728" y="430.56" style="fill: #ff0000; stroke: #ff0000"/>
     <use xlink:href="#m256b22af28" x="895.999432" y="420.780492" style="fill: #ff0000; stroke: #ff0000"/>
    </g>
   </g>
   <g id="line2d_16">
    <path d="M 84.363068 193.311713 
L 127.080772 133.920281 
L 169.798475 84.857793 
L 212.516178 62.822711 
L 255.233882 52.493766 
L 297.951585 44.574908 
L 340.669288 42.33697 
L 383.386992 37.516796 
L 426.104695 34.76241 
L 468.822398 32.868771 
L 511.540102 33.901665 
L 554.257805 30.630833 
L 596.975508 28.048596 
L 639.693212 29.25364 
L 682.410915 28.392894 
L 725.128618 30.114385 
L 767.846322 30.286534 
L 810.564025 27.36 
L 853.281728 27.36 
L 895.999432 29.942236 
" clip-path="url(#pa4ece0553c)" style="fill: none; stroke: #0000ff; stroke-width: 1.5; stroke-linecap: square"/>
    <defs>
     <path id="m772fa5f452" d="M -3 3 
L 3 -3 
M -3 -3 
L 3 3 
" style="stroke: #0000ff"/>
    </defs>
    <g clip-path="url(#pa4ece0553c)">
     <use xlink:href="#m772fa5f452" x="84.363068" y="193.311713" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="127.080772" y="133.920281" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="169.798475" y="84.857793" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="212.516178" y="62.822711" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="255.233882" y="52.493766" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="297.951585" y="44.574908" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="340.669288" y="42.33697" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="383.386992" y="37.516796" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="426.104695" y="34.76241" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="468.822398" y="32.868771" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="511.540102" y="33.901665" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="554.257805" y="30.630833" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="596.975508" y="28.048596" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="639.693212" y="29.25364" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="682.410915" y="28.392894" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="725.128618" y="30.114385" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="767.846322" y="30.286534" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="810.564025" y="27.36" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="853.281728" y="27.36" style="fill: #0000ff; stroke: #0000ff"/>
     <use xlink:href="#m772fa5f452" x="895.999432" y="29.942236" style="fill: #0000ff; stroke: #0000ff"/>
    </g>
   </g>
   <g id="line2d_17">
    <path d="M 84.363068 156.59513 
L 127.080772 178.429128 
L 169.798475 299.643574 
L 212.516178 196.590149 
L 255.233882 346.452297 
L 297.951585 335.390783 
L 340.669288 346.666697 
L 383.386992 314.166294 
L 426.104695 391.224574 
L 468.822398 408.068172 
L 511.540102 342.098293 
L 554.257805 405.989254 
L 596.975508 375.959147 
L 639.693212 416.426121 
L 682.410915 397.845521 
L 725.128618 402.282865 
L 767.846322 418.941631 
L 810.564025 424.572639 
L 853.281728 417.222854 
L 895.999432 427.151142 
" clip-path="url(#pa4ece0553c)" style="fill: none; stroke: #008000; stroke-width: 1.5; stroke-linecap: square"/>
    <defs>
     <path id="md3cf578cd7" d="M 0 3 
C 0.795609 3 1.55874 2.683901 2.12132 2.12132 
C 2.683901 1.55874 3 0.795609 3 0 
C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 
C 1.55874 -2.683901 0.795609 -3 0 -3 
C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 
C -2.683901 -1.55874 -3 -0.795609 -3 0 
C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 
C -1.55874 2.683901 -0.795609 3 0 3 
z
" style="stroke: #008000"/>
    </defs>
    <g clip-path="url(#pa4ece0553c)">
     <use xlink:href="#md3cf578cd7" x="84.363068" y="156.59513" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="127.080772" y="178.429128" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="169.798475" y="299.643574" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="212.516178" y="196.590149" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="255.233882" y="346.452297" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="297.951585" y="335.390783" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="340.669288" y="346.666697" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="383.386992" y="314.166294" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="426.104695" y="391.224574" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="468.822398" y="408.068172" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="511.540102" y="342.098293" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="554.257805" y="405.989254" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="596.975508" y="375.959147" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="639.693212" y="416.426121" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="682.410915" y="397.845521" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="725.128618" y="402.282865" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="767.846322" y="418.941631" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="810.564025" y="424.572639" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="853.281728" y="417.222854" style="fill: #008000; stroke: #008000"/>
     <use xlink:href="#md3cf578cd7" x="895.999432" y="427.151142" style="fill: #008000; stroke: #008000"/>
    </g>
   </g>
   <g id="line2d_18">
    <path d="M 84.363068 218.637813 
L 127.080772 120.333479 
L 169.798475 66.025491 
L 212.516178 163.642382 
L 255.233882 58.463619 
L 297.951585 67.400377 
L 340.669288 67.400377 
L 383.386992 95.585535 
L 426.104695 42.652432 
L 468.822398 33.715675 
L 511.540102 64.650605 
L 554.257805 35.778004 
L 596.975508 48.151976 
L 639.693212 29.591017 
L 682.410915 39.215218 
L 725.128618 38.527775 
L 767.846322 30.965903 
L 810.564025 28.216132 
L 853.281728 32.340789 
L 895.999432 28.216132 
" clip-path="url(#pa4ece0553c)" style="fill: none; stroke: #bfbf00; stroke-width: 1.5; stroke-linecap: square"/>
    <defs>
     <path id="m3b0205920d" d="M -3 3 
L 3 -3 
M -3 -3 
L 3 3 
" style="stroke: #bfbf00"/>
    </defs>
    <g clip-path="url(#pa4ece0553c)">
     <use xlink:href="#m3b0205920d" x="84.363068" y="218.637813" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="127.080772" y="120.333479" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="169.798475" y="66.025491" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="212.516178" y="163.642382" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="255.233882" y="58.463619" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="297.951585" y="67.400377" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="340.669288" y="67.400377" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="383.386992" y="95.585535" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="426.104695" y="42.652432" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="468.822398" y="33.715675" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="511.540102" y="64.650605" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="554.257805" y="35.778004" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="596.975508" y="48.151976" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="639.693212" y="29.591017" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="682.410915" y="39.215218" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="725.128618" y="38.527775" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="767.846322" y="30.965903" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="810.564025" y="28.216132" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="853.281728" y="32.340789" style="fill: #bfbf00; stroke: #bfbf00"/>
     <use xlink:href="#m3b0205920d" x="895.999432" y="28.216132" style="fill: #bfbf00; stroke: #bfbf00"/>
    </g>
   </g>
   <g id="patch_3">
    <path d="M 43.78125 450.72 
L 43.78125 7.2 
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 936.58125 450.72 
L 936.58125 7.2 
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 43.78125 450.72 
L 936.58125 450.72 
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 43.78125 7.2 
L 936.58125 7.2 
" style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="legend_1">
    <g id="patch_7">
     <path d="M 50.78125 445.72 
L 180.996875 445.72 
Q 182.996875 445.72 182.996875 443.72 
L 182.996875 386.0075 
Q 182.996875 384.0075 180.996875 384.0075 
L 50.78125 384.0075 
Q 48.78125 384.0075 48.78125 386.0075 
L 48.78125 443.72 
Q 48.78125 445.72 50.78125 445.72 
z
" style="fill: #ffffff; opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
    </g>
    <g id="line2d_19">
     <path d="M 52.78125 392.105937 
L 62.78125 392.105937 
L 72.78125 392.105937 
" style="fill: none; stroke: #ff0000; stroke-width: 1.5; stroke-linecap: square"/>
     <g>
      <use xlink:href="#m256b22af28" x="62.78125" y="392.105937" style="fill: #ff0000; stroke: #ff0000"/>
     </g>
    </g>
    <g id="text_17">
     <!-- Training Loss -->
     <g transform="translate(80.78125 395.605937) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-54" d="M -19 4666 
L 3928 4666 
L 3928 4134 
L 2272 4134 
L 2272 0 
L 1638 0 
L 1638 4134 
L -19 4134 
L -19 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-69" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-67" d="M 2906 1791 
Q 2906 2416 2648 2759 
Q 2391 3103 1925 3103 
Q 1463 3103 1205 2759 
Q 947 2416 947 1791 
Q 947 1169 1205 825 
Q 1463 481 1925 481 
Q 2391 481 2648 825 
Q 2906 1169 2906 1791 
z
M 3481 434 
Q 3481 -459 3084 -895 
Q 2688 -1331 1869 -1331 
Q 1566 -1331 1297 -1286 
Q 1028 -1241 775 -1147 
L 775 -588 
Q 1028 -725 1275 -790 
Q 1522 -856 1778 -856 
Q 2344 -856 2625 -561 
Q 2906 -266 2906 331 
L 2906 616 
Q 2728 306 2450 153 
Q 2172 0 1784 0 
Q 1141 0 747 490 
Q 353 981 353 1791 
Q 353 2603 747 3093 
Q 1141 3584 1784 3584 
Q 2172 3584 2450 3431 
Q 2728 3278 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 434 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-54"/>
      <use xlink:href="#DejaVuSans-72" x="46.333984"/>
      <use xlink:href="#DejaVuSans-61" x="87.447266"/>
      <use xlink:href="#DejaVuSans-69" x="148.726562"/>
      <use xlink:href="#DejaVuSans-6e" x="176.509766"/>
      <use xlink:href="#DejaVuSans-69" x="239.888672"/>
      <use xlink:href="#DejaVuSans-6e" x="267.671875"/>
      <use xlink:href="#DejaVuSans-67" x="331.050781"/>
      <use xlink:href="#DejaVuSans-20" x="394.527344"/>
      <use xlink:href="#DejaVuSans-4c" x="426.314453"/>
      <use xlink:href="#DejaVuSans-6f" x="480.277344"/>
      <use xlink:href="#DejaVuSans-73" x="541.458984"/>
      <use xlink:href="#DejaVuSans-73" x="593.558594"/>
     </g>
    </g>
    <g id="line2d_20">
     <path d="M 52.78125 406.784062 
L 62.78125 406.784062 
L 72.78125 406.784062 
" style="fill: none; stroke: #0000ff; stroke-width: 1.5; stroke-linecap: square"/>
     <g>
      <use xlink:href="#m772fa5f452" x="62.78125" y="406.784062" style="fill: #0000ff; stroke: #0000ff"/>
     </g>
    </g>
    <g id="text_18">
     <!-- Training Accuracy -->
     <g transform="translate(80.78125 410.284062) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-54"/>
      <use xlink:href="#DejaVuSans-72" x="46.333984"/>
      <use xlink:href="#DejaVuSans-61" x="87.447266"/>
      <use xlink:href="#DejaVuSans-69" x="148.726562"/>
      <use xlink:href="#DejaVuSans-6e" x="176.509766"/>
      <use xlink:href="#DejaVuSans-69" x="239.888672"/>
      <use xlink:href="#DejaVuSans-6e" x="267.671875"/>
      <use xlink:href="#DejaVuSans-67" x="331.050781"/>
      <use xlink:href="#DejaVuSans-20" x="394.527344"/>
      <use xlink:href="#DejaVuSans-41" x="426.314453"/>
      <use xlink:href="#DejaVuSans-63" x="492.972656"/>
      <use xlink:href="#DejaVuSans-63" x="547.953125"/>
      <use xlink:href="#DejaVuSans-75" x="602.933594"/>
      <use xlink:href="#DejaVuSans-72" x="666.3125"/>
      <use xlink:href="#DejaVuSans-61" x="707.425781"/>
      <use xlink:href="#DejaVuSans-63" x="768.705078"/>
      <use xlink:href="#DejaVuSans-79" x="823.685547"/>
     </g>
    </g>
    <g id="line2d_21">
     <path d="M 52.78125 421.462187 
L 62.78125 421.462187 
L 72.78125 421.462187 
" style="fill: none; stroke: #008000; stroke-width: 1.5; stroke-linecap: square"/>
     <g>
      <use xlink:href="#md3cf578cd7" x="62.78125" y="421.462187" style="fill: #008000; stroke: #008000"/>
     </g>
    </g>
    <g id="text_19">
     <!-- Validation Loss -->
     <g transform="translate(80.78125 424.962187) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-56" d="M 1831 0 
L 50 4666 
L 709 4666 
L 2188 738 
L 3669 4666 
L 4325 4666 
L 2547 0 
L 1831 0 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6c" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-74" d="M 1172 4494 
L 1172 3500 
L 2356 3500 
L 2356 3053 
L 1172 3053 
L 1172 1153 
Q 1172 725 1289 603 
Q 1406 481 1766 481 
L 2356 481 
L 2356 0 
L 1766 0 
Q 1100 0 847 248 
Q 594 497 594 1153 
L 594 3053 
L 172 3053 
L 172 3500 
L 594 3500 
L 594 4494 
L 1172 4494 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-56"/>
      <use xlink:href="#DejaVuSans-61" x="60.658203"/>
      <use xlink:href="#DejaVuSans-6c" x="121.9375"/>
      <use xlink:href="#DejaVuSans-69" x="149.720703"/>
      <use xlink:href="#DejaVuSans-64" x="177.503906"/>
      <use xlink:href="#DejaVuSans-61" x="240.980469"/>
      <use xlink:href="#DejaVuSans-74" x="302.259766"/>
      <use xlink:href="#DejaVuSans-69" x="341.46875"/>
      <use xlink:href="#DejaVuSans-6f" x="369.251953"/>
      <use xlink:href="#DejaVuSans-6e" x="430.433594"/>
      <use xlink:href="#DejaVuSans-20" x="493.8125"/>
      <use xlink:href="#DejaVuSans-4c" x="525.599609"/>
      <use xlink:href="#DejaVuSans-6f" x="579.5625"/>
      <use xlink:href="#DejaVuSans-73" x="640.744141"/>
      <use xlink:href="#DejaVuSans-73" x="692.84375"/>
     </g>
    </g>
    <g id="line2d_22">
     <path d="M 52.78125 436.140312 
L 62.78125 436.140312 
L 72.78125 436.140312 
" style="fill: none; stroke: #bfbf00; stroke-width: 1.5; stroke-linecap: square"/>
     <g>
      <use xlink:href="#m3b0205920d" x="62.78125" y="436.140312" style="fill: #bfbf00; stroke: #bfbf00"/>
     </g>
    </g>
    <g id="text_20">
     <!-- Validation Accuracy -->
     <g transform="translate(80.78125 439.640312) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-56"/>
      <use xlink:href="#DejaVuSans-61" x="60.658203"/>
      <use xlink:href="#DejaVuSans-6c" x="121.9375"/>
      <use xlink:href="#DejaVuSans-69" x="149.720703"/>
      <use xlink:href="#DejaVuSans-64" x="177.503906"/>
      <use xlink:href="#DejaVuSans-61" x="240.980469"/>
      <use xlink:href="#DejaVuSans-74" x="302.259766"/>
      <use xlink:href="#DejaVuSans-69" x="341.46875"/>
      <use xlink:href="#DejaVuSans-6f" x="369.251953"/>
      <use xlink:href="#DejaVuSans-6e" x="430.433594"/>
      <use xlink:href="#DejaVuSans-20" x="493.8125"/>
      <use xlink:href="#DejaVuSans-41" x="525.599609"/>
      <use xlink:href="#DejaVuSans-63" x="592.257812"/>
      <use xlink:href="#DejaVuSans-63" x="647.238281"/>
      <use xlink:href="#DejaVuSans-75" x="702.21875"/>
      <use xlink:href="#DejaVuSans-72" x="765.597656"/>
      <use xlink:href="#DejaVuSans-61" x="806.710938"/>
      <use xlink:href="#DejaVuSans-63" x="867.990234"/>
      <use xlink:href="#DejaVuSans-79" x="922.970703"/>
     </g>
    </g>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="pa4ece0553c">
   <rect x="43.78125" y="7.2" width="892.8" height="443.52"/>
  </clipPath>
 </defs>
</svg>
 output.svg…]()




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
