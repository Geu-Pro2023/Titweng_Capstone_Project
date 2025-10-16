import 'package:flutter/material.dart';
import 'package:smooth_page_indicator/smooth_page_indicator.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:titweng/utils/app_theme.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _controller = PageController();
  int _currentPage = 0;

  final List<Map<String, dynamic>> _pages = [
    {
      'image': 'assets/images/onboarding/1.png',
      'title': 'Protect Your Heritage',
      'description':
          'Secure your cattle wealth with advanced biometric nose print technology',
      'color': AppTheme.primaryColor,
    },
    {
      'image': 'assets/images/onboarding/2.png',
      'title': 'Instant Verification',
      'description':
          'Scan any cattle in seconds to verify ownership and prevent illegal market transactions',
      'color': AppTheme.secondaryColor,
    },
    {
      'image': 'assets/images/onboarding/3.png',
      'title': 'Community Trust',
      'description':
          'Build a transparent ecosystem where buyers and sellers can trade with confidence',
      'color': AppTheme.accentColor,
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Gradient
          Container(
            decoration: AppTheme.gradientBackground,
          ),

          // Page View
          PageView.builder(
            controller: _controller,
            itemCount: _pages.length,
            onPageChanged: (index) => setState(() => _currentPage = index),
            itemBuilder: (context, index) {
              return OnboardingPage(page: _pages[index]);
            },
          ),

          // Top Section
          Positioned(
            top: 60,
            left: 24,
            right: 24,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Logo
                Container(
                  width: 50,
                  height: 50,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.asset(
                      'assets/images/logo.png',
                      width: 50,
                      height: 50,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),

                // Skip Button
                if (_currentPage != _pages.length - 1)
                  TextButton(
                    onPressed: () => _controller.jumpToPage(_pages.length - 1),
                    child: Text(
                      'Skip',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.8),
                        fontSize: 16,
                      ),
                    ),
                  ),
              ],
            ),
          ),

          // Bottom Section
          Positioned(
            bottom: 40,
            left: 24,
            right: 24,
            child: Column(
              children: [
                // Indicator
                SmoothPageIndicator(
                  controller: _controller,
                  count: _pages.length,
                  effect: ExpandingDotsEffect(
                    activeDotColor: Colors.white,
                    dotColor: Colors.white.withOpacity(0.3),
                    dotHeight: 8,
                    dotWidth: 8,
                    spacing: 8,
                    expansionFactor: 3,
                  ),
                ),

                const SizedBox(height: 32),

                // Action Button
                _currentPage == _pages.length - 1
                    ? _buildGetStartedButton()
                    : _buildNextButton(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNextButton() {
    return FloatingActionButton(
      backgroundColor: Colors.white,
      onPressed: () => _controller.nextPage(
        duration: 500.ms,
        curve: Curves.easeInOut,
      ),
      child: Icon(
        Icons.arrow_forward,
        color: AppTheme.primaryColor,
        size: 28,
      ),
    ).animate().slide(delay: 300.ms, begin: Offset(0, 1), end: Offset.zero);
  }

  Widget _buildGetStartedButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: () => Navigator.pushReplacementNamed(context, '/login'),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.white,
          foregroundColor: AppTheme.primaryColor,
          padding: const EdgeInsets.symmetric(vertical: 18),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          elevation: 4,
        ),
        child: Text(
          'Get Started',
          style: AppTheme.buttonText.copyWith(color: AppTheme.primaryColor),
        ),
      ),
    ).animate().fadeIn(delay: 200.ms);
  }
}

class OnboardingPage extends StatelessWidget {
  final Map<String, dynamic> page;

  const OnboardingPage({super.key, required this.page});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Image
          Container(
            width: 280,
            height: 280,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.3),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Image.asset(
                page['image'],
                fit: BoxFit.cover,
              ),
            ),
          )
              .animate()
              .scale(duration: 600.ms, curve: Curves.easeOutBack)
              .fadeIn(duration: 400.ms),

          const SizedBox(height: 48),

          // Title
          Text(
            page['title'],
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.w700,
              color: Colors.white,
              height: 1.2,
            ),
            textAlign: TextAlign.center,
          )
              .animate()
              .fadeIn(delay: 200.ms)
              .slideY(begin: 0.5, end: 0, duration: 600.ms),

          const SizedBox(height: 16),

          // Description
          Text(
            page['description'],
            style: TextStyle(
              fontSize: 16,
              color: Colors.white.withOpacity(0.9),
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          )
              .animate()
              .fadeIn(delay: 400.ms)
              .slideY(begin: 0.3, end: 0, duration: 600.ms),
        ],
      ),
    );
  }
}
