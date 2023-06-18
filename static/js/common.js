// common.js

window.onload = function() {
  var nav = document.querySelector('nav');
  nav.style.float = 'left';
  nav.style.width = '200px';
  nav.style.height = '100%';
  nav.style.backgroundColor = '#f2f2f2';
  nav.style.position = 'fixed';
  nav.style.top = '0';

  var welcome = document.querySelector('.welcome');
  welcome.style.textAlign = 'center';
  welcome.style.paddingTop = '50px';
}


// 点击导航栏中的链接后，将当前链接的文本设置为红色
var links = document.querySelectorAll('nav ul li a');
for (var i = 0; i < links.length; i++) {
  links[i].addEventListener('click', function() {
    this.style.color = 'red';
  });
}

// 为页面中的所有图片添加点击事件，当用户点击图片时，在控制台输出图片的 URL
var images = document.querySelectorAll('img');
for (var i = 0; i < images.length; i++) {
  images[i].addEventListener('click', function() {
    console.log(this.src);
  });
}

// 在页面加载完毕后，弹出欢迎框
window.addEventListener('load', function() {
  alert('欢迎来到我的网站！');
});
