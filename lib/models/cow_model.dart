class CowModel {
  final String id;
  final String name;
  final int age;
  final String ownerName;
  final String ownerEmail;
  final String location;
  final String state;
  final String witness;
  final List<String> noseImagePaths;
  final DateTime registrationDate;
  final String additionalDetails;

  CowModel({
    required this.id,
    required this.name,
    required this.age,
    required this.ownerName,
    required this.ownerEmail,
    required this.location,
    required this.state,
    required this.witness,
    required this.noseImagePaths,
    required this.registrationDate,
    this.additionalDetails = '',
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'age': age,
      'owner_name': ownerName,
      'owner_email': ownerEmail,
      'location': location,
      'state': state,
      'witness': witness,
      'nose_image_paths': noseImagePaths,
      'registration_date': registrationDate.toIso8601String(),
      'additional_details': additionalDetails,
    };
  }

  factory CowModel.fromJson(Map<String, dynamic> json) {
    return CowModel(
      id: json['id'],
      name: json['name'],
      age: json['age'],
      ownerName: json['owner_name'],
      ownerEmail: json['owner_email'],
      location: json['location'],
      state: json['state'],
      witness: json['witness'],
      noseImagePaths: List<String>.from(json['nose_image_paths']),
      registrationDate: DateTime.parse(json['registration_date']),
      additionalDetails: json['additional_details'] ?? '',
    );
  }
}