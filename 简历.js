// 文件名：script.js
document.addEventListener('DOMContentLoaded', () => {
    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            target.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // 动态时间轴指示器
    document.querySelectorAll('.timeline-item').forEach((item, index) => {
        item.style.animation = `fadeInUp ${index * 0.2}s`;
    });
});