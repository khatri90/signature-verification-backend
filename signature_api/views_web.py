# In signature_api/views_web.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from django.conf import settings

from .models import UserProfile, Signature, VerificationRecord
from .utils.verification_handler import SignatureVerificationHandler

import os
from django.core.files.base import ContentFile

@login_required
def dashboard(request):
    context = {
        'profile_count': UserProfile.objects.count(),
        'signature_count': Signature.objects.count(),
        'verification_count': VerificationRecord.objects.count(),
        'recent_profiles': UserProfile.objects.order_by('-created_at')[:5],
        'recent_verifications': VerificationRecord.objects.order_by('-verified_at')[:5]
    }
    return render(request, 'base/dashboard.html', context)

# User Profile views
@login_required
def profile_list(request):
    profiles = UserProfile.objects.all().order_by('-created_at')
    return render(request, 'profiles/profile_list.html', {'profiles': profiles})

@login_required
def profile_detail(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk)
    return render(request, 'profiles/profile_detail.html', {'profile': profile})

@login_required
def profile_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        id_number = request.POST.get('id_number')
        
        # Simple validation
        if not name or not id_number:
            messages.error(request, "Both name and ID number are required")
            return render(request, 'profiles/profile_form.html', {
                'title': 'Create User Profile',
                'form': {'name': {'value': name}, 'id_number': {'value': id_number}}
            })
        
        # Check if ID number already exists
        if UserProfile.objects.filter(id_number=id_number).exists():
            messages.error(request, "A profile with this ID number already exists")
            return render(request, 'profiles/profile_form.html', {
                'title': 'Create User Profile',
                'form': {'name': {'value': name}, 'id_number': {'value': id_number}, 
                         'id_number': {'errors': ['ID number must be unique']}}
            })
        
        # Create profile
        profile = UserProfile.objects.create(
            name=name,
            id_number=id_number,
            created_by=request.user
        )
        
        messages.success(request, f"Profile for {name} created successfully")
        return redirect('profile_detail', pk=profile.id)
    
    return render(request, 'profiles/profile_form.html', {'title': 'Create User Profile'})

@login_required
def profile_edit(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk)
    
    if request.method == 'POST':
        name = request.POST.get('name')
        id_number = request.POST.get('id_number')
        
        # Simple validation
        if not name or not id_number:
            messages.error(request, "Both name and ID number are required")
            return render(request, 'profiles/profile_form.html', {
                'title': 'Edit User Profile',
                'form': {'name': {'value': name}, 'id_number': {'value': id_number}}
            })
        
        # Check if ID number already exists for a different profile
        if UserProfile.objects.filter(id_number=id_number).exclude(pk=profile.pk).exists():
            messages.error(request, "A profile with this ID number already exists")
            return render(request, 'profiles/profile_form.html', {
                'title': 'Edit User Profile',
                'form': {'name': {'value': name}, 'id_number': {'value': id_number}, 
                         'id_number': {'errors': ['ID number must be unique']}}
            })
        
        # Update profile
        profile.name = name
        profile.id_number = id_number
        profile.save()
        
        messages.success(request, f"Profile for {name} updated successfully")
        return redirect('profile_detail', pk=profile.id)
    
    return render(request, 'profiles/profile_form.html', {
        'title': 'Edit User Profile',
        'form': {'name': {'value': profile.name}, 'id_number': {'value': profile.id_number}}
    })

@login_required
def profile_delete(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk)
    
    if request.method == 'POST':
        profile_name = profile.name
        profile.delete()
        messages.success(request, f"Profile for {profile_name} deleted successfully")
        return redirect('profile_list')
    
    return render(request, 'profiles/profile_delete.html', {'profile': profile})

# Signature views
@login_required
def signature_add(request, profile_id):
    profile = get_object_or_404(UserProfile, pk=profile_id)
    
    if request.method == 'POST':
        if 'image' not in request.FILES:
            messages.error(request, "Please select a signature image to upload")
            return render(request, 'signatures/signature_form.html', {'profile': profile})
        
        image = request.FILES['image']
        notes = request.POST.get('notes', '')
        
        # Initialize verifier for preprocessing
        verifier = SignatureVerificationHandler()
        
        # Preprocess the image before saving
        processed_image_path = verifier.preprocess_image(image)
        
        # Read the processed image
        with open(processed_image_path, 'rb') as f:
            processed_content = f.read()
        
        # Create a ContentFile with the processed image
        processed_file = ContentFile(processed_content, name=image.name)
        
        # Create signature with processed image
        signature = Signature.objects.create(
            user_profile=profile,
            added_by=request.user,
            notes=notes
        )
        signature.image.save(image.name, processed_file)
        
        # Clean up temporary file
        os.remove(processed_image_path)
        
        messages.success(request, f"Signature added successfully to {profile.name}'s profile")
        return redirect('profile_detail', pk=profile.id)
    
    return render(request, 'signatures/signature_form.html', {'profile': profile})

@login_required
def signature_delete(request, pk):
    signature = get_object_or_404(Signature, pk=pk)
    profile = signature.user_profile
    
    if request.method == 'POST':
        signature.delete()
        messages.success(request, "Signature deleted successfully")
        return redirect('profile_detail', pk=profile.id)
    
    return render(request, 'signatures/signature_delete.html', {'signature': signature})

# Verification views
@login_required
def verification_form(request):
    profiles = UserProfile.objects.all().order_by('name')
    selected_profile = request.GET.get('profile_id')
    
    context = {
        'profiles': profiles,
        'selected_profile': selected_profile
    }
    
    if request.method == 'POST':
        user_profile_id = request.POST.get('user_profile_id')
        save_to_references = 'save_to_references' in request.POST
        notes = request.POST.get('notes', '')
        
        # Get user profile
        try:
            user_profile = UserProfile.objects.get(pk=user_profile_id)
        except UserProfile.DoesNotExist:
            messages.error(request, "Please select a valid user profile")
            return render(request, 'verification/verification_form.html', context)
        
        # Check if test signature is present
        if 'test_signature' not in request.FILES:
            messages.error(request, "Please select a signature image to verify")
            return render(request, 'verification/verification_form.html', context)
        
        test_signature = request.FILES['test_signature']
        
        # Get reference signatures
        reference_signatures = []
        for signature in user_profile.signatures.all():
            reference_signatures.append(signature.image.path)
        
        if not reference_signatures:
            messages.error(request, f"No reference signatures found for {user_profile.name}. Please add reference signatures first.")
            return render(request, 'verification/verification_form.html', context)
        
        # Initialize verifier
        verifier = SignatureVerificationHandler()
        
        # Save the test signature temporarily
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_test_path = os.path.join(temp_dir, f"test_{test_signature.name}")
        
        with open(temp_test_path, 'wb+') as destination:
            for chunk in test_signature.chunks():
                destination.write(chunk)
        
        # Preprocess the test signature
        processed_test_path = verifier.preprocess_image(test_signature)
        
        # Verify the signature
        verification_results = verifier.verify_signature(processed_test_path, reference_signatures)
        
        # Create verification record
        verification_record = VerificationRecord(
            user_profile=user_profile,
            result='genuine' if verification_results['is_genuine'] else 'forged',
            confidence=verification_results['confidence'] * 100,  # Convert to percentage
            verified_by=request.user,
            verification_metrics=verification_results['metrics']
        )
        
        # Save the test signature to the record
        verification_record.test_signature.save(
            f"test_{test_signature.name}", 
            ContentFile(open(processed_test_path, 'rb').read())
        )
        
        verification_record.save()
        
        # If the signature is genuine and user wants to save it as a reference
        if verification_results['is_genuine'] and save_to_references:
            new_signature = Signature(
                user_profile=user_profile,
                added_by=request.user,
                notes=notes
            )
            new_signature.image.save(
                test_signature.name,
                ContentFile(open(processed_test_path, 'rb').read())
            )
            verification_record.added_to_references = True
            verification_record.save()
        
        # Clean up
        os.remove(temp_test_path)
        os.remove(processed_test_path)
        
        # Add result to context
        context['result'] = verification_record
        
        messages.success(request, f"Verification complete. Result: {verification_record.result.title()}")
        
    return render(request, 'verification/verification_form.html', context)

@login_required
def verification_history(request):
    records = VerificationRecord.objects.all().order_by('-verified_at')
    return render(request, 'verification/verification_history.html', {'records': records})

@login_required
def verification_detail(request, pk):
    record = get_object_or_404(VerificationRecord, pk=pk)
    return render(request, 'verification/verification_detail.html', {'record': record})

@login_required
def add_to_references(request, pk):
    verification = get_object_or_404(VerificationRecord, pk=pk)
    
    # Check if already added
    if verification.added_to_references:
        messages.error(request, "This signature is already added to references")
        return redirect('verification_detail', pk=verification.id)
    
    # Check if it's genuine
    if verification.result != 'genuine':
        messages.error(request, "Only genuine signatures can be added to references")
        return redirect('verification_detail', pk=verification.id)
    
    # Create a new signature record
    new_signature = Signature(
        user_profile=verification.user_profile,
        added_by=request.user,
        notes=request.POST.get('notes', 'Added from verification')
    )
    
    # Copy the test signature to the signature record
    new_signature.image.save(
        os.path.basename(verification.test_signature.name),
        verification.test_signature.file
    )
    
    # Update verification record
    verification.added_to_references = True
    verification.save()
    
    messages.success(request, "Signature added to references successfully")
    return redirect('verification_detail', pk=verification.id)

# Authentication views
class CustomLoginView(LoginView):
    template_name = 'base/login.html'
    redirect_authenticated_user = True