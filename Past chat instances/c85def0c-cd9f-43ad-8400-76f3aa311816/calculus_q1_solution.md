# Giải Tích - Câu 1: Đổi thứ tự lấy tích phân

**Đề bài:** Đổi thứ tự lấy tích phân sau:
$$I = \int_0^1 dx \int_{\sqrt{x/3}}^{\sqrt{x}} f(x, y)dy + \int_1^{4/3} dx \int_{\sqrt{x/3}}^{2-x} f(x, y)dy$$

---

### Bước 1: Xác định miền lấy tích phân $D$ ban đầu
Miền $D$ được chia làm hai phần theo trục x:

**Phần 1:**
- $0 \le x \le 1$
- $\sqrt{\frac{x}{3}} \le y \le \sqrt{x}$

**Phần 2:**
- $1 \le x \le \frac{4}{3}$
- $\sqrt{\frac{x}{3}} \le y \le 2 - x$

---

### Bước 2: Tìm phương trình các đường biên theo biến $y$
Để đổi thứ tự lấy tích phân (từ $dx\,dy$ sang $dy\,dx$), ta cần chiếu miền $D$ lên **trục y**. Do đó, ta phải viết lại phương trình các đường biên dưới dạng $x = g(y)$:

1. **Đường biên dưới (chung cho cả 2 phần):**
   $$y = \sqrt{\frac{x}{3}} \implies y^2 = \frac{x}{3} \implies x = 3y^2$$
2. **Đường biên trên (của Phần 1):**
   $$y = \sqrt{x} \implies x = y^2$$
3. **Đường biên trên (của Phần 2):**
   $$y = 2 - x \implies x = 2 - y$$

*(Lưu ý: Vì $x \ge 0$ trong toàn bộ miền nên $y$ cũng luôn $\ge 0$.)*

---

### Bước 3: Tìm các giao điểm (Giới hạn của $y$)
Ta cần tìm giá trị nhỏ nhất, lớn nhất của $y$ và các điểm mà đường biên thay đổi. 

Tìm tung độ ($y$) của các giao điểm:
- **Giao điểm của $y = \sqrt{x}$ và $y = \sqrt{x/3}$**: Gặp nhau tại $(0, 0)$. Vậy min $y = 0$.
- **Giao điểm của $y = \sqrt{x}$ và $x = 1$**: Gặp nhau tại $(1, 1)$. Vậy max $y = 1$.
- **Giao điểm của $y = \sqrt{x/3}$ và $y = 2 - x$**: 
  $$\sqrt{\frac{x}{3}} = 2 - x \implies \frac{x}{3} = (2-x)^2 = 4 - 4x + x^2$$
  Nhân chéo với 3 và chuyển vế ta được phương trình bậc hai: $3x^2 - 13x + 12 = 0$.
  Giải ra ta được $x = 3$ (loại vì nằm ngoài miền) và **$x = \frac{4}{3}$**.
  Thay $x = \frac{4}{3}$ vào $y = 2 - x$, ta được **$y = \frac{2}{3}$**.

---

### Bước 4: Chia lại miền theo trục y
Bây giờ ta quét miền theo chiều ngang (nhìn từ trục y):

- **Biên trái** (x nhỏ hơn) LUÔN LUÔN là đường $x = y^2$.
- **Biên phải** (x lớn hơn) bị thay đổi tùy theo độ cao của $y$:
   - Từ $y = 0$ đến $y = \frac{2}{3}$: Biên phải là đường cong $x = 3y^2$.
   - Từ $y = \frac{2}{3}$ đến $y = 1$: Biên phải là đường thẳng $x = 2 - y$.

---

### Bước 5: Viết tích phân mới
Vì biên phải bị gãy tại $y = \frac{2}{3}$, ta phải tách tích phân mới làm 2 phần dọc theo trục y:

**Phần dưới:** $y$ chạy từ $0$ đến $\frac{2}{3}$, tương ứng $x$ chạy từ $y^2$ đến $3y^2$.
**Phần trên:** $y$ chạy từ $\frac{2}{3}$ đến $1$, tương ứng $x$ chạy từ $y^2$ đến $2-y$.

Vậy, đáp án cuối cùng là:
$$I = \int_0^{2/3} dy \int_{y^2}^{3y^2} f(x, y) dx + \int_{2/3}^1 dy \int_{y^2}^{2-y} f(x, y) dx$$
