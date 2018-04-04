#include "align.h"
#include <string>
#include <cstdlib>
#include <limits>

using std::string;
using std::cout;
using std::endl;

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{	
	int n_rows = (srcImage.n_rows / 3);
	int n_cols = srcImage.n_cols;

	Image blue = srcImage.submatrix(0, 0, n_rows, n_cols);
	Image green = srcImage.submatrix(n_rows, 0, n_rows, n_cols);
	Image red = srcImage.submatrix(2 * n_rows, 0, n_rows, n_cols);
	float min_mse1 = std::numeric_limits<float>::infinity();
	float min_mse2 = std::numeric_limits<float>::infinity();
	int row1 = 0;
	int col1 = 0;
	int row2 = 0;
	int col2 = 0;

	for (int row = -15; row < 16; ++row){
		for (int col = -15; col < 16; ++col){
			int start_i = row < 0 ? -row : 0;
			int start_j = col < 0 ? -col : 0;
			float mse1 = 0;
			float mse2 = 0;

			for (int i = start_i + 0.05 * n_rows; i < 0.95 * n_rows - (start_i + row); ++i){
				for (int j = start_j + 0.05 * n_cols; j < 0.95 * n_cols - (start_j + col); ++j){
					int g = std::get<0>(green(i, j));
					int r = std::get<0>(red(i + row, j + col));
					int b = std::get<0>(blue(i + row, j + col));
					mse1 += (g - r) * (g - r);
					mse2 += (g - b) * (g - b);
				}
			}
			mse1 /= (0.9 * n_rows - 2 * std::abs(row)) * (0.9 * n_cols - 2 * std::abs(col));
			mse2 /= (0.9 * n_rows - 2 * std::abs(row)) * (0.9 * n_cols - 2 * std::abs(col));

			if (mse1 < min_mse1){
				min_mse1 = mse1;
				row1 = row;
				col1 = col;
			}
			if (mse2 < min_mse2){
				min_mse2 = mse2;
				row2 = row;
				col2 = col;
			}
		}
	}

	Image new_im(n_rows, n_cols);
	for (int i = 0; i < n_rows; ++i){
		for (int j = 0; j < n_cols; ++j){
			uint g = std::get<0>(green(i, j));
			uint r = (i + row1 >= 0) && (i + row1 < n_rows) && (j + col1 >= 0) && (j + col1 < n_cols) ?
					 std::get<0>(red(i + row1, j + col1)) : 0;
			uint b = (i + row2 >= 0) && (i + row2 < n_rows) && (j + col2 >= 0) && (j + col2 < n_cols) ?
					 std::get<0>(blue(i + row2, j + col2)) : 0;
			new_im(i, j) = std::make_tuple(r, g, b);
		}
	}
	srcImage = new_im;

	if (isPostprocessing){
		if (postprocessingType == "--gray-world") return gray_world(src_image);
		if (postprocessingType == "--unsharp") return unsharp(src_image);
		if (postprocessingType == "autocontrast") return autocontrast(src_image, fraction);
	}
    return srcImage;
}

Image mirror(Image src_image, uint add_rows, uint add_cols){
	Image new_im(src_image.n_rows + 2 * add_rows, src_image.n_cols + 2 * add_cols);

	for (uint i = 0; i < src_image.n_rows; ++i){
		for (uint j = 0; j < src_image.n_cols; ++j){
			new_im(i + add_rows, j + add_cols) = src_image(i, j);
		}
	}

	for (uint i = 1; i <= add_rows; ++i){
		for (uint j = 0; j < src_image.n_cols; ++j){
			new_im(add_rows - i, j + add_cols) = src_image(i - 1, j);
			new_im(src_image.n_rows + add_rows - 1 + i, j) = src_image(src_image.n_rows - i, j);
		}
	}

	for (uint i = 0; i < src_image.n_rows; ++i){
		for (uint j = 1; j <= add_cols; ++j){
			new_im(i + add_rows, add_cols - j) = src_image(i, j - 1);
			new_im(i, src_image.n_cols + add_cols -1 + j) = src_image(i, src_image.n_cols - j);
		}
	}

	for (uint i = 0; i < add_rows; ++i){
		for (uint j = 0; j < add_cols; ++j){
			new_im(i, j) = src_image(add_rows - 1 - i, add_cols - 1 - j);
			new_im(src_image.n_rows + add_rows + i, j) = src_image(src_image.n_rows - 1 - i, add_cols - 1 - j);
			new_im(i, src_image.n_cols + add_cols + j) = src_image(add_rows - 1 - i, src_image.n_cols - 1 - j);
			new_im(src_image.n_rows + add_rows + i, src_image.n_cols + add_cols + j) = src_image(src_image.n_rows - 1 - i, src_image.n_cols - 1 - j);
		}
	}

	return new_im;
}

Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel);
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}

Image unsharp(Image src_image) {
	Matrix<double> kernel = {{-1.0/6, -2.0/3, -1.0/6},
							 {-2.0/3, 13.0/3, -2.0/3},
							 {-1.0/6, -2.0/3, -1.0/6}};
    return custom(src_image, kernel);
}

Image gray_world(Image src_image) {
	double mean_red = 0;
	double mean_green = 0;
	double mean_blue = 0;
	double mean_gray = 0;
	int n_rows = src_image.n_rows;
	int n_cols = src_image.n_cols;

	for (int i = 0.05 * n_rows; i < 0.95 * n_rows; ++i){
		for (int j = 0.05 * n_cols; j < 0.95 * n_cols; ++j){
			uint r, g, b;
			std::tie(r, g, b) = src_image(i, j);
			mean_red += r;
			mean_green += g;
			mean_blue += b;
		}
	}
	mean_red /= (0.81 * n_rows * n_cols);
	mean_green /= (0.81 * n_rows * n_cols);
	mean_blue /= (0.81 * n_rows * n_cols);
	mean_gray = (mean_red + mean_green + mean_blue) / 3;

	double norm_red = mean_gray / mean_red;
	double norm_green = mean_gray / mean_green;
	double norm_blue = mean_gray / mean_blue;

	for (int i = 0.05 * n_rows; i < 0.95 * n_rows; ++i){
		for (int j = 0.05 * n_cols; j < 0.95 * n_cols; ++j){
			uint r, g, b;
			std::tie(r, g, b) = src_image(i, j);
			r *= norm_red;
			g *= norm_green;
			b *= norm_blue;
			r = r > 255 ? 255 : r;
			g = g > 255 ? 255 : g;
			b = b > 255 ? 255 : b;
			src_image(i, j) = std::make_tuple(r, g, b);
		}
	}	
    return src_image;
}

Image resize(Image src_image, double scale) {
    return src_image;
}

Image custom(Image src_image, Matrix<double> kernel) {
    // Function custom is useful for making concrete linear filtrations
    // like gaussian or sobel. So, we assume that you implement custom
    // and then implement other filtrations using this function.
    // sobel_x and sobel_y are given as an example.

	Image big_im = mirror(src_image, kernel.n_rows / 2, kernel.n_cols / 2);
    Image new_im(big_im.n_rows, big_im.n_cols);
    new_im = big_im.deep_copy();
    for (uint m = kernel.n_rows / 2; m < big_im.n_rows - (kernel.n_rows / 2); ++m){
    	for (uint n = kernel.n_cols / 2; n < big_im.n_cols - (kernel.n_cols / 2); ++n){
    		float red = 0;
    		float green = 0;
    		float blue = 0;
    		for (uint i = 0; i < kernel.n_rows; ++i){
    			for (uint j = 0; j < kernel.n_cols; ++j){
    				double r, g, b;
    				std::tie(r, g, b) = big_im(m + i -(kernel.n_rows / 2), n + j - (kernel.n_cols / 2));
    				red += r * kernel(i, j);
    				green += g * kernel(i, j);
    				blue += b * kernel(i, j);
    			}
    		}
            if (red >= 256) red = 255;
            if (green >= 256) green = 255;
            if (blue >= 256) blue = 255;

            if (red < 0) red = 0;
            if (green < 0) green = 0;
            if (blue < 0) blue = 0;
    		new_im(m, n) = std::make_tuple(static_cast<int>(red), static_cast<int>(green), static_cast<int>(blue));
    	}
    }
    src_image = new_im.submatrix(kernel.n_rows / 2, kernel.n_cols / 2, src_image.n_rows, src_image.n_cols);
    return src_image;
}

Image autocontrast(Image src_image, double fraction) {
	int distr[256] = {0};
	int pxl_to_miss = fraction * src_image.n_rows * src_image.n_cols;
	for (uint i = 0; i < src_image.n_rows; ++i){
		for (uint j = 0; j < src_image.n_cols; ++j){
			uint r, g, b;
			std::tie(r, g, b) = src_image(i, j);
			uint y = 0.2125 * r + 0.7154 * g + 0.0721 * b;
			distr[y]++;
		}
	}

	int min = 0;
	int max = 255;
	int sum = 0;
	for(uint i = 0; i < 256; ++i){
		sum += distr[i];
		if (sum >= pxl_to_miss){
			min = i;
			break;
		}
	}
	sum = 0;
	for(int i = 255; i >= 0; --i){
		sum += distr[i];
		if (sum >= pxl_to_miss){
			max = i;
			break;
		}
	}

	for (uint i = 0; i < src_image.n_rows; ++i){
		for (uint j = 0; j < src_image.n_cols; ++j){
			int r, g, b;
			std::tie(r, g, b) = src_image(i, j);
			r = 255 * (r - min) / (max - min);
			g = 255 * (g - min) / (max - min);
			b = 255 * (b - min) / (max - min);
			r = r > 255 ? 255 : r;
			g = g > 255 ? 255 : g;
			b = b > 255 ? 255 : b;
			r = r < 0 ? 0 : r;
			g = g < 0 ? 0 : g;
			b = b < 0 ? 0 : b;
			src_image(i, j) = std::make_tuple(r, g, b);
		}
	}
    return src_image;
}

Image gaussian(Image src_image, double sigma, int radius)  {
    return src_image;
}

Image gaussian_separable(Image src_image, double sigma, int radius) {
    return src_image;
}

Image median(Image src_image, int radius) {
	if (radius == 0) return src_image;
	uint rad = radius;
	Image big_im = mirror(src_image, rad, rad);
	Image new_im(big_im.n_rows, big_im.n_cols);
	new_im = big_im.deep_copy();
	uint count = (2 * rad + 1) * (2 * rad + 1) / 2;
	for (uint m = rad; m < big_im.n_rows - rad; ++m){
		for (uint n = rad; n < big_im.n_cols - rad; ++n){
			uint distr_r[256] = {0};
			uint distr_g[256] = {0};
			uint distr_b[256] = {0};
    		for (uint i = 0; i < 2 * rad + 1; ++i){
    			for (uint j = 0; j < 2 * rad + 1; ++j){
    				uint r, g, b;
    				std::tie(r, g, b) = big_im(m + i - rad, n + j - rad);
    				if (m == 40 && n ==40) std::cout << r << ' ';
    				distr_r[r]++;
    				distr_g[g]++;
    				distr_b[b]++;
    			}
    		}
			uint sum_r = 0;
			uint sum_g = 0;
			uint sum_b = 0;
    		uint med_r = 0;
    		uint med_g = 0;
    		uint med_b = 0;
    		bool flag_r = true;
    		bool flag_g = true;
    		bool flag_b = true;
    		for (uint i = 0; i < 256; ++i){
    			sum_r += distr_r[i];
    			sum_g += distr_g[i];
    			sum_b += distr_b[i];
    			if ((sum_r > count) && flag_r) {med_r = i; flag_r = false;}
    			if ((sum_g > count) && flag_g) {med_g = i; flag_g = false;}
    			if ((sum_b > count) && flag_b) {med_b = i; flag_b = false;}
    			if (!(flag_r || flag_g || flag_b)) break;
    		}
    		if (m == 40 && n ==40) std::cout << std::endl << med_r;
    		new_im(m, n) = std::make_tuple(med_r, med_g, med_b);
		}
	}
	src_image = new_im.submatrix(radius, radius, src_image.n_rows, src_image.n_cols);
    return src_image;
}

Image median_linear(Image src_image, int radius) {
    return src_image;
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

Image canny(Image src_image, int threshold1, int threshold2) {
    return src_image;
}
