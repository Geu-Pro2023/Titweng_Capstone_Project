import 'package:flutter/material.dart';

class AppTheme {
  // Dark Green Theme
  static const Color primaryColor = Color(0xFF044440); // Dark Green
  static const Color secondaryColor = Color(0xFF044440); // Dark Green
  static const Color accentColor = Color(0xFF044440); // Dark Green
  static const Color backgroundColor = Color(0xFF044440); // Dark Green background
  static const Color surfaceColor = Color(0xFFFFFFFF);
  static const Color cardColor = Color(0xFFF1F8E9); // Very light green
  static const Color textColor = Color(0xFF1A1A1A); // Dark text
  static const Color lightTextColor = Color(0xFF666666); // Gray text
  static const Color successColor = Color(0xFF2E7D32); // Dark Forest Green
  static const Color warningColor = Color(0xFFE65100); // Dark Orange
  static const Color errorColor = Color(0xFFC62828); // Dark Red
  static const Color infoColor = Color(0xFF1565C0); // Dark Blue
  
  // System Colors
  static const Color verifiedColor = Color(0xFF2E7D32); // Dark Forest Green
  static const Color suspiciousColor = Color(0xFFE65100); // Dark Orange
  static const Color pendingColor = Color(0xFFE65100); // Dark Orange
  static const Color adminColor = Color(0xFF6A4C93); // Purple
  static const Color userColor = Color(0xFF40916C); // Medium Green
  
  // Navigation Colors
  static const Color navBarColor = Color(0xFFFFFFFF); // White
  static const Color navSelectedColor = Color(0xFF044440); // Dark Green
  static const Color navUnselectedColor = Color(0xFF044440); // Dark Green
  
  // Gradient Colors
  static const List<Color> primaryGradient = [Color(0xFF044440), Color(0xFF044440)];
  static const List<Color> adminGradient = [Color(0xFF044440), Color(0xFF044440)];
  static const List<Color> userGradient = [Color(0xFF044440), Color(0xFF044440)];
  static const List<Color> successGradient = [Color(0xFF2E7D32), Color(0xFF4CAF50)];
  static const List<Color> warningGradient = [Color(0xFFE65100), Color(0xFFFF9800)];

  // Text Styles (using default fonts)
  static TextStyle headlineLarge = TextStyle(
    fontSize: 32,
    fontWeight: FontWeight.w700,
    color: textColor,
    height: 1.2,
  );

  static TextStyle headlineMedium = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.w600,
    color: textColor,
  );

  static TextStyle titleLarge = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w600,
    color: textColor,
  );

  static TextStyle bodyLarge = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    color: textColor,
    height: 1.5,
  );

  static TextStyle bodySmall = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    color: lightTextColor,
  );

  static TextStyle buttonText = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w600,
    color: Colors.white,
  );

  // Theme Data
  static ThemeData get theme {
    return ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.light(
        primary: primaryColor,
        secondary: secondaryColor,
        surface: surfaceColor,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: textColor,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        titleTextStyle: headlineMedium.copyWith(color: primaryColor),
        iconTheme: const IconThemeData(color: primaryColor),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 32),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          elevation: 2,
          textStyle: buttonText,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceColor,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide.none,
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        hintStyle: bodyLarge.copyWith(color: lightTextColor),
        labelStyle: bodyLarge.copyWith(color: primaryColor),
      ),
      cardTheme: CardThemeData(
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        margin: EdgeInsets.zero,
        shadowColor: Colors.black.withValues(alpha: 0.1),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: navBarColor,
        selectedItemColor: navSelectedColor,
        unselectedItemColor: navUnselectedColor,
        type: BottomNavigationBarType.fixed,
        elevation: 8,
        selectedLabelStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 12),
        unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w500, fontSize: 11),
      ),
    );
  }

  // Custom Box Decorations
  static BoxDecoration get cardDecoration {
    return BoxDecoration(
      color: surfaceColor,
      borderRadius: BorderRadius.circular(16),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withValues(alpha: 0.08),
          blurRadius: 20,
          offset: const Offset(0, 4),
          spreadRadius: 0,
        ),
      ],
    );
  }

  static BoxDecoration get gradientBackground {
    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: primaryGradient,
      ),
    );
  }
  
  static BoxDecoration get adminGradientDecoration {
    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: adminGradient,
      ),
      borderRadius: BorderRadius.circular(20),
    );
  }
  
  static BoxDecoration get userGradientDecoration {
    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: userGradient,
      ),
      borderRadius: BorderRadius.circular(20),
    );
  }
  
  static BoxDecoration get glassMorphism {
    return BoxDecoration(
      color: Colors.white.withValues(alpha: 0.1),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(
        color: Colors.white.withValues(alpha: 0.2),
        width: 1,
      ),
    );
  }
}
