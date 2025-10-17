import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:titweng/firebase_options.dart';
import 'package:titweng/utils/app_theme.dart';
import 'package:titweng/screens/splash_screen.dart';
import 'package:titweng/screens/onboarding_screen.dart';
import 'package:titweng/screens/auth_screen.dart';
import 'package:titweng/screens/main_dashboard.dart';
import 'package:titweng/models/user_model.dart';
import 'package:titweng/screens/cattle_details_screen.dart';
import 'package:titweng/screens/verification_screen.dart';
import 'package:titweng/screens/report_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const TitwengApp());
}

class TitwengApp extends StatelessWidget {
  const TitwengApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Titweng - Cattle Verification System',
      theme: AppTheme.theme,
      debugShowCheckedModeBanner: false,
      initialRoute: '/splash',
      routes: {
        '/splash': (context) => const SplashScreen(),
        '/onboarding': (context) => const OnboardingScreen(),
        '/login': (context) => const AuthScreen(),
        '/main-dashboard': (context) => MainDashboard(
              user: ModalRoute.of(context)!.settings.arguments as UserModel,
            ),
        '/cattle-details': (context) => CattleDetailsScreen(
              cattle: ModalRoute.of(context)!.settings.arguments as dynamic,
            ),
        '/verification': (context) => const VerificationScreen(),
        '/report': (context) => const ReportScreen(),
      },
    );
  }
}
