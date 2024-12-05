document.addEventListener("DOMContentLoaded", function () {
	const videos = document.querySelectorAll(".results-carousel .video-section");
	const prevButton = document.getElementById("prev-button");
	const nextButton = document.getElementById("next-button");
	let currentIndex = 0;
  
	// 初期状態で最初の動画をアクティブにする
	videos[currentIndex].classList.add("active");
  
	// 次の動画に切り替える
	nextButton.addEventListener("click", function () {
	  videos[currentIndex].classList.remove("active");
	  currentIndex = (currentIndex + 1) % videos.length; // 次のインデックス（ループ対応）
	  videos[currentIndex].classList.add("active");
	});
  
	// 前の動画に切り替える
	prevButton.addEventListener("click", function () {
	  videos[currentIndex].classList.remove("active");
	  currentIndex = (currentIndex - 1 + videos.length) % videos.length; // 前のインデックス（ループ対応）
	  videos[currentIndex].classList.add("active");
	});
  });
  